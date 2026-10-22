"""Verify the entire frozen cylinder matrix, identities, and saved-trace scores."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import mujoco
import numpy as np
from run_orca_suite_job import summarize

OUT = ROOT / "artifacts/orca_cylinder_study"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(allow_partial=False):
    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, expected in protocol["frozen_inputs"].items():
        assert digest(ROOT / path) == expected, path
    missing, train_count, eval_count, total, recomputed = [], 0, 0, 0, 0
    paired = {}
    for phase in ("train", "eval"):
        for path in protocol[phase+"_jobs"]:
            job = json.loads((ROOT / path).read_text())
            assert digest(ROOT / path) == protocol["job_sha256"][path]
            run = ROOT / job["output"]
            if not (run / "summary.json").exists():
                missing.append(job["output"])
                continue
            summary = json.loads((run / "summary.json").read_text())
            config = json.loads((run / "run_config.json").read_text())
            assert summary["job_sha256"] == config["job_sha256"] == digest(ROOT / path)
            assert summary["actual_steps"] == job["steps"]
            for key, value in job.items():
                assert config[key] == value, (path, key)
            for name, expected in config["source_sha256"].items():
                assert digest(run / "sources" / name) == expected
                assert digest(ROOT / "scripts" / name) == expected
            assert digest(run / "reference.npz") == protocol["frozen_inputs"][protocol["reference"]]
            assert digest(run / "model.xml") == config["model_sha256"]
            if job.get("checkpoint"):
                assert digest(ROOT / job["checkpoint"]) == config["checkpoint_sha256"]
            model = mujoco.MjModel.from_xml_path(str(run / "model.xml"))
            assert model.nu == 17 and model.geom("cube").type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
            np.testing.assert_allclose(model.geom("cube").size[:2], [.0265, .0265])
            np.testing.assert_allclose(model.body("cube").mass, .035)
            env = SimpleNamespace(arm=config["arm"], residual_bound=.08,
                                  _lowers=model.actuator_ctrlrange[:, 0], _uppers=model.actuator_ctrlrange[:, 1])
            with (run / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            expected_steps = ([0, 163840, 327680, 491520, 655360] if phase == "train" else [0])
            assert [int(r["step"]) for r in rows] == expected_steps
            for row in rows:
                folder = run / f"step_{int(row['step']):09d}"
                episodes = json.loads((folder / "episodes.json").read_text())
                seeds = list(range(job["eval_start"], job["eval_start"]+job["eval_count"]))
                assert [ep["seed"] for ep in episodes] == seeds
                assert np.isclose(float(row["clean_success30"]), np.mean([e["clean_success30"] for e in episodes]))
                for ep in episodes:
                    expected = ep["success30"] and not ep["substep_forbidden"] and ep["post_release_penetration_m"] <= .003
                    assert ep["clean_success30"] == expected
                with np.load(folder / "initial_states.npz", allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["seeds"], seeds)
                    key = (phase, job.get("benchmark", "B0_nominal"), tuple(seeds))
                    if key in paired:
                        np.testing.assert_array_equal(paired[key], z["qpos"])
                    else:
                        paired[key] = z["qpos"].copy()
                with np.load(folder / "preview_trajectories.npz", allow_pickle=False) as z:
                    trace = {k:z[k] for k in z.files}
                assert all(np.isfinite(v).all() for v in trace.values())
                _, preview = summarize(trace, env, seeds[:4])
                for left, right in zip(preview, episodes[:4]):
                    for name in ("success30", "clean_success30", "final_valid", "substep_forbidden"):
                        assert left[name] == right[name]
                    for name in ("longest_hold_sec", "post_release_penetration_m", "residual_rms_rad"):
                        assert np.isclose(left[name], right[name], atol=1e-6)
                if config["arm"] == "residual":
                    assert np.max(np.abs(trace["effective_residual"])) <= .08001
                else:
                    np.testing.assert_array_equal(trace["video_base"], 0.)
                if job["method"] == "zero":
                    np.testing.assert_allclose(trace["ctrl"], trace["video_base"], atol=1e-5)
                if job.get("benchmark") == "B5_push":
                    np.testing.assert_allclose(np.linalg.norm(trace["push"][:, 200:204], axis=-1), .15, atol=1e-6)
                recomputed += len(preview)
            if phase == "train":
                train_count += 1
            else:
                eval_count += 1
                total += len(episodes)
    complete = not missing
    if complete:
        assert (train_count, eval_count, total) == (9, 66, 4224)
    result = dict(complete=complete, missing=missing, training_runs=train_count, evaluation_jobs=eval_count,
                  test_episodes=total, saved_preview_episodes_recomputed=recomputed,
                  note="All episode records/identities checked; state-trace recomputation covers four saved previews per evaluation, not all episodes",
                  frozen_inputs_checked=len(protocol["frozen_inputs"]),
                  protocol_sha256=digest(OUT / "protocol.json"))
    (OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    if missing and not allow_partial:
        raise RuntimeError(f"Incomplete cylinder study: {len(missing)} missing jobs")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-partial", action="store_true")
    audit(p.parse_args().allow_partial)
