"""Verify S02 identities, paired starts, stored rollout metrics and coverage."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_preview(trace, episodes, config):
    valid = (trace["alive"] & (trace["thumb"] > 0) & (trace["contacts"] >= 2)
             & (trace["forbidden"] == 0) & (trace["released"] > 0)
             & (trace["cube_z"] > trace["spawn_z"] - .01))
    count = np.zeros(len(episodes), int)
    best = count.copy()
    for t in range(800):
        count = np.where(valid[:, t], count+1, 0)
        best = np.maximum(count, best)
    holds = best * .05
    np.testing.assert_allclose(holds, trace["hold"].max(axis=1), atol=1e-5)
    success = (holds >= 30) & valid[:, -1]
    post = trace["active"] & (trace["released"] > 0)
    depth = np.max(np.where(post, trace["substep_penetration"], 0), axis=1)
    forbidden = np.any((trace["substep_forbidden"] > 0) & post, axis=1)
    clean = success & ~forbidden & (depth <= .003)
    for i, row in enumerate(episodes):
        assert bool(success[i]) == row["success30"]
        assert bool(clean[i]) == row["clean_success30"]
        np.testing.assert_allclose(holds[i], row["longest_hold_sec"], atol=1e-5)
        np.testing.assert_allclose(depth[i], row["post_release_penetration_m"], atol=1e-9)
        active = trace["active"][i]
        np.testing.assert_allclose(trace["substep_force_peak"][i][active].max(), row["substep_force_peak_n"])
    assert np.max(np.abs(np.diff(trace["ctrl"], axis=1))) <= .10001
    if config["arm"] == "residual":
        gain = config.get("residual_gain", 1.)
        np.testing.assert_allclose(trace["action"], trace["policy_action"]*gain, atol=1e-6)
        np.testing.assert_allclose(trace["requested_residual"], trace["action"]*.08, atol=1e-6)
        np.testing.assert_allclose(trace["effective_residual"], trace["ctrl"]-trace["video_base"], atol=1e-6)
        assert np.max(np.abs(trace["requested_residual"])) <= .08001
        if config["method"] == "zero":
            assert np.max(np.abs(trace["requested_residual"])) == 0
    if config["benchmark"] == "B5_push":
        magnitudes = np.linalg.norm(trace["push"], axis=-1)
        expected = np.zeros(800)
        expected[200:204] = .15
        np.testing.assert_allclose(magnitudes, np.broadcast_to(expected, magnitudes.shape), atol=1e-6)
    else:
        assert np.max(np.abs(trace["push"])) == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--require-videos", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    frozen = protocol["source_hashes"]
    for path, expected in frozen.items():
        assert digest(ROOT / path) == expected, path
    starts, completed, missing = {}, [], []
    training, evaluations, previews = 0, 0, 0
    for record in protocol["jobs"]:
        job_path = ROOT / record["path"]
        assert digest(job_path) == record["sha256"], record["name"]
        job = json.loads(job_path.read_text())
        run = root / record["name"]
        if not (run / "summary.json").exists():
            missing.append(record["name"])
            continue
        summary = json.loads((run / "summary.json").read_text())
        config = json.loads((run / "run_config.json").read_text())
        assert summary["status"] == "completed"
        assert summary["job_sha256"] == config["job_sha256"] == record["sha256"]
        for key, value in job.items():
            assert config[key] == value, (record["name"], key)
        for name, expected in config["source_sha256"].items():
            assert digest(run / "sources" / name) == expected
            if "scripts/"+name in frozen:
                assert expected == frozen["scripts/"+name]
        for key, path in (("model_sha256", config["model"]), ("reference_sha256", config["reference"])):
            assert digest(Path(path)) == config[key]
        with (run / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        values = [float(row["success30"]) for row in rows]
        assert summary["first"] == values[0] and summary["last"] == values[-1]
        assert summary["max"] == max(values)
        assert summary["best_step"] == int(rows[values.index(max(values))]["step"])
        if job["steps"]:
            training += 1
            assert summary["final_step"] == job["steps"] == 163840 and len(rows) == 5
            assert all(row["clean_success30"] == "" for row in rows)
            assert (run / "params.pkl").stat().st_size > 1000
        else:
            assert job["audit"] and summary["final_step"] == 0 and len(rows) == 1
            if job.get("checkpoint"):
                assert config["checkpoint_sha256"] == digest(ROOT / job["checkpoint"])
            step = run / "step_000000000"
            episodes = json.loads((step / "episodes.json").read_text())
            assert [r["seed"] for r in episodes] == protocol["test_seeds"]
            assert all(isinstance(r["clean_success30"], bool) for r in episodes)
            assert all(not r["clean_success30"] or r["success30"] for r in episodes)
            assert all(np.isfinite(r["max_penetration_m"]) and r["max_penetration_m"] >= 0 for r in episodes)
            np.testing.assert_allclose(float(rows[0]["success30"]), np.mean([r["success30"] for r in episodes]))
            np.testing.assert_allclose(float(rows[0]["clean_success30"]), np.mean([r["clean_success30"] for r in episodes]))
            with np.load(step / "initial_states.npz") as z:
                np.testing.assert_array_equal(z["seeds"], protocol["test_seeds"])
                qpos = z["qpos"].copy()
                assert np.isfinite(qpos).all()
            benchmark = job["benchmark"]
            if benchmark in starts:
                np.testing.assert_array_equal(starts[benchmark], qpos)
            else:
                starts[benchmark] = qpos
            with np.load(step / "preview_trajectories.npz") as z:
                trace = {key: z[key] for key in z.files}
            assert trace["qpos"].shape[:2] == (4, 800)
            assert all(np.isfinite(v).all() for v in trace.values())
            check_preview(trace, episodes[:4], config)
            previews += 4
            evaluations += len(episodes)
        completed.append(record["name"])
    videos = None
    if args.require_videos:
        videos = json.loads((root / "videos/manifest.json").read_text())
        assert videos["complete"] and len(videos["cases"]) == 30
        for case in videos["cases"]:
            expected_seed = None if case["method"].startswith("zero_") else 1001
            assert case["train_seed"] == expected_seed and case["test_seed"] == 40000
            for video in case["videos"].values():
                assert digest(root / video["path"]) == video["sha256"]
    if not missing:
        assert training == 50 and evaluations == 77600 and previews == 1552
        assert len(starts) == 6
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "jax", "jaxlib", "mujoco", "brax", "flax", "optax")}
    result = dict(complete=not missing, completed_jobs=len(completed), missing=missing,
                  training=training, evaluated_episodes=evaluations, independently_recomputed_preview_episodes=previews,
                  scope="All completed job identities, all episode summary aggregates, all paired reset states; independent time-series metric recomputation covers first four saved cases per job only.",
                  versions=versions, videos_checked=videos is not None,
                  protocol_sha256=digest(root / "protocol.json"),
                  statistics_plan_sha256=digest(root / "statistics_plan.json"),
                  audit_source_sha256=digest(Path(__file__)))
    (root / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "missing"}))
    if missing and not args.allow_partial:
        raise RuntimeError(f"Incomplete matrix: {len(missing)} missing jobs")


if __name__ == "__main__":
    main()
