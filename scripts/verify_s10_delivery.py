"""Check S10 provenance, completed budgets, paired initial states and video files."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_s10_study import check
from s10_mobile_base import OUT, ROOT
from retarget_s09_reference import digest


def read(path):
    return json.loads(path.read_text())


def verify():
    protocol = check()
    completed = []
    shared_sources = {}

    def verify_sources(run, config):
        for source, sha in config["sources"].items():
            assert digest(run / "sources" / source) == sha
            if source in shared_sources:
                assert shared_sources[source] == sha, f"Source drift across runs: {source}"
            shared_sources[source] = sha
    for seed in protocol["seeds"]:
        for method in ("ppo", "guided", "residual"):
            run = OUT / "train" / f"{method}_s{seed}"
            summary = read(run / "summary.json")
            assert summary["status"] == "completed"
            assert summary["final_step"] == protocol["steps"]
            with (run / "metrics.csv").open() as handle:
                metrics = list(csv.DictReader(handle))
            assert [int(row["step"]) for row in metrics] == [0, 40960, 81920, 122880, 163840]
            assert digest(run / "params.pkl") == digest(run / "step_000163840/params.pkl")
            config = read(run / "run_config.json")
            assert config["seed"] == seed and config["method"] == method
            verify_sources(run, config)
            completed.append(str(run.relative_to(ROOT)))

    expected_seeds = np.arange(protocol["test"][0], sum(protocol["test"]))
    baseline_qpos = None
    evaluated = []
    tasks = [("reference", 0), ("reference_exact_quality", 0)]
    tasks += [(m, s) for s in protocol["seeds"] for m in ("ppo", "bc", "guided", "residual")]
    for method, seed in tasks:
        run = OUT / "eval" / f"{method}_s{seed}"
        config = read(run / "run_config.json")
        verify_sources(run, config)
        summary = read(run / "summary.json")
        assert summary["status"] == "completed" and summary["final_step"] == 0
        folder = run / "step_000000000"
        rows = read(folder / "episodes.json")
        assert [r["seed"] for r in rows] == expected_seeds.tolist()
        with np.load(folder / "initial_states.npz") as initial:
            np.testing.assert_array_equal(initial["seeds"], expected_seeds)
            if baseline_qpos is None:
                baseline_qpos = initial["qpos"].copy()
            np.testing.assert_array_equal(initial["qpos"], baseline_qpos)
        if method not in ("reference", "reference_exact_quality"):
            train_method = "guided" if method == "bc" else method
            step = 0 if method == "bc" else protocol["steps"]
            source = OUT / "train" / f"{train_method}_s{seed}" / f"step_{step:09d}/params.pkl"
            assert Path(config["checkpoint"]).resolve() == source.resolve()
            assert config["checkpoint_sha256"] == digest(source)
            assert digest(folder / "params.pkl") == digest(source)
        for row in rows:
            assert row["released_rope_force_n"] == 0
            assert row["hand_travel_m"] > .001, "No measurable physical base movement"
            assert row["full_path_success30"] == (row["success30"] and row["max_path_penetration_m"] <= .003)
            if row["success30"]:
                assert row["hold_sec"] >= 30 and row["final_valid"]
                assert not row["forbidden"]
                assert row["initial_penetration_m"] <= .001
                assert row["max_released_penetration_m"] <= .003
            if method == "reference_exact_quality":
                assert row["residual_rms_rad"] == 0
        evaluated.append(dict(run=str(run.relative_to(ROOT)), count=len(rows), successes=sum(r["success30"] for r in rows)))

    collection = read(OUT / "demonstrations/dataset_manifest.json")
    assert digest(OUT / "demonstrations/trajectories.npz") == collection["dataset_sha256"]
    assert not set(collection["attempted_seeds"]) & set(expected_seeds)
    with (OUT / "report/episodes.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == len(tasks) * len(expected_seeds)
    videos = read(OUT / "videos/manifest.json")
    assert len(videos) == 14
    for row in videos:
        path = OUT / "videos" / row["file"]
        assert digest(path) == row["sha256"] and row["decoded"]
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True, capture_output=True)
    for name in ("demonstrations_gpu", "bc_s22101_native", "residual_s22101_native",
                 "ppo_s22102_native", "guided_s22102_native", "guided_s22103_native", "residual_s22103_native"):
        assert read(OUT / "audits" / f"{name}.json")["passed"]
    for name in ("reference_exact_policy_v2", "bc_s22101_policy", "guided_s22101_policy", "residual_s22101_policy"):
        assert read(OUT / "audits" / f"{name}.json")["status"] == "verified"
    clearance = read(OUT / "audits/forearm_previews_final/results.json")
    assert clearance["status"] == "clear_in_checked_frames"
    checked = {(r["run"], r["seed"]): r for r in clearance["episodes"]}
    expected_previews = set()
    for trace in (OUT / "eval").glob("*/step_000000000/preview_trajectories.npz"):
        with np.load(trace) as saved:
            for seed in saved["seeds"]:
                key = (trace.parent.parent.name, int(seed))
                expected_previews.add(key)
                assert checked[key]["trace_sha256"] == digest(trace)
                assert checked[key]["min_forearm_hull_clearance_m"] > 0
    assert set(checked) == expected_previews
    for file in ("current.json", "evaluation_current.json"):
        assert read(OUT / "execution" / file)["status"] == "terminal"
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "scripts", "-p", "test_s10*.py"],
                           cwd=ROOT, env=dict(os.environ, JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                           capture_output=True, text=True)
    assert tests.returncode == 0, tests.stdout + tests.stderr
    result = dict(status="verified", frozen_files=len(protocol["hashes"]), training_runs=completed,
                  evaluations=evaluated, unique_policy_state_evaluations=len(tasks)*len(expected_seeds),
                  video_files=len(videos), paired_initial_states="bitwise equal",
                  forearm_preview_records=len(checked),
                  forearm_min_sampled_clearance_m=min(r["min_forearm_hull_clearance_m"] for r in clearance["episodes"]),
                  tests=dict(returncode=tests.returncode, stdout=tests.stdout, stderr=tests.stderr),
                  scope="All final policy budgets/provenance/outcomes; physics replay only saved audited previews, not every trajectory")
    (OUT / "report/completion_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    verify()
