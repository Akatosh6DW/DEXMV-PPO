"""Re-score every confirmatory trace and check the complete registered matrix."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from analyze_orca_residual import curve_summary
from evaluate_orca_residual_suite import verify_sources
from orca_residual_controls import load_reference
from orca_residual_metrics import summarize

ROOT = Path(__file__).resolve().parents[1]


def audit_run(run, traces):
    meta = json.loads((run/"run_config.json").read_text())
    verify_sources(run, meta)
    for name, expected in meta["source_sha256"].items():
        archived = run/"sources"/name
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == expected, f"Archived source changed: {archived}"
    assert hashlib.sha256((run/"model.xml").read_bytes()).hexdigest() == meta["model_sha256"]
    model = mujoco.MjModel.from_xml_path(str(run/"model.xml"))
    low, high = model.actuator_ctrlrange.T
    cfg = json.loads((run/"env_config.json").read_text())
    reference = None
    if meta["arm"] == "residual":
        assert hashlib.sha256((run/"reference.npz").read_bytes()).hexdigest() == meta["reference_sha256"]
        reference, _ = load_reference(run/"reference.npz", low, high, cfg["initial_ctrl"],
                                       .05, 800, cfg["max_ctrl_delta"])
        with np.load(run/"processed_reference.npz", allow_pickle=False) as z:
            np.testing.assert_allclose(reference, z["ctrl"], atol=1e-6)
    with (run/"metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    training_metrics = run/"ppo_metrics.jsonl"
    if training_metrics.exists():
        for line in training_metrics.read_text().splitlines():
            for key, value in json.loads(line).items():
                assert np.isfinite(value), f"Non-finite training metric: {run}: {key}"
    for row in rows:
        assert all(np.isfinite(float(value)) for value in row.values()), f"Non-finite evaluation metric: {run}"
        checkpoint = run/f"step_{int(row['step']):09d}"
        path = checkpoint/"trajectories.npz"
        with np.load(path, allow_pickle=False) as z:
            trace = dict(z)
        for key, value in trace.items():
            assert np.isfinite(value).all(), f"Non-finite trace field: {path}: {key}"
        assert trace["ctrl"].shape == (meta["eval_count"], 800, 17)
        assert trace["obs"].shape == (meta["eval_count"], 800, 113)
        assert np.isfinite(trace["qpos"]).all() and np.isfinite(trace["qvel"]).all()
        height_index = int(model.joint("cube_freejoint").qposadr[0])+2
        velocity_index = int(model.joint("cube_freejoint").dofadr[0])+2
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # MJX step() calls forward(), then integrates. Contact/xpos caches are
        # pre-integration; reconstruct that free-joint position exactly.
        cached_height = trace["qpos"][:, :, height_index]-model.opt.timestep*trace["qvel"][:, :, velocity_index]
        np.testing.assert_allclose(trace["cube_z"], cached_height, atol=1e-6)
        if reference is not None:
            np.testing.assert_allclose(trace["video_base"],
                                       np.broadcast_to(reference, trace["video_base"].shape), atol=1e-6)
        else:
            np.testing.assert_array_equal(trace["obs"][:, :, -18:-1], 0)
        actual, holds, success = summarize(trace, meta["arm"], meta["bound"], low, high,
                                           cfg["max_ctrl_delta"])
        np.testing.assert_array_less(np.abs(trace["ctrl"][:, 0]-np.array(cfg["initial_ctrl"])),
                                     cfg["max_ctrl_delta"]+1e-5)
        for key, value in actual.items():
            np.testing.assert_allclose(value, float(row[key]), atol=1e-5, err_msg=f"{run}: {key}")
        episodes = json.loads((checkpoint/"episodes.json").read_text())
        assert episodes["seeds"] == list(range(meta["eval_start"], meta["eval_start"]+meta["eval_count"]))
        np.testing.assert_allclose(episodes["hold_sec"], holds, atol=1e-5)
        np.testing.assert_array_equal(episodes["success30"], success)
        if meta["zero"]:
            assert meta["arm"] == "residual"
            np.testing.assert_allclose(trace["action"], 0, atol=1e-7)
            np.testing.assert_allclose(trace["ctrl"], trace["video_base"], atol=1e-5)
        traces[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        traces[str(checkpoint/"params.pkl")] = hashlib.sha256((checkpoint/"params.pkl").read_bytes()).hexdigest()
    return meta, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--baseline-runs", type=Path, nargs=3, required=True)
    p.add_argument("--repeat-runs", type=Path, nargs="+", required=True)
    p.add_argument("--candidate-repeat-run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((ROOT/"artifacts/orca_residual/confirmatory_protocol.json").read_text())
    analysis = json.loads((args.suite/"analysis.json").read_text())
    expected = {(shape, arm, seed) for shape in protocol["tasks"]
                for arm in protocol["methods"] for seed in protocol["training_seeds"]}
    actual = {(r["task"], r["arm"], r["seed"]) for r in analysis["runs"]}
    assert actual == expected and len(analysis["runs"]) == len(expected)
    assert json.loads((args.suite/"completed.json").read_text())["completed"]
    traces, configs, models, checkpoints = {}, {}, {}, {}
    runtime_versions = set()
    dataset_sha = hashlib.sha256((ROOT/protocol["bc_dataset"]).read_bytes()).hexdigest()
    for record in analysis["runs"]:
        run = Path(record["run"])
        meta, rows = audit_run(run, traces)
        assert meta["confirmatory"] and not meta["development"]
        assert meta["backend"] == "gpu"
        runtime_versions.add((meta["jax_version"], meta["mujoco_version"], meta["backend"]))
        assert meta["steps"] == protocol["ppo_steps_per_run"]
        assert meta["evals"] == protocol["evals_per_run"]
        assert meta["eval_start"] == protocol["validation_seeds"][0]
        assert meta["eval_count"] == len(protocol["validation_seeds"])
        assert meta["learning_rate"] == protocol["learning_rate"]
        assert meta["init_std"] == protocol["init_pre_tanh_std"]
        assert meta["bound"] == protocol["residual_bound_rad"]
        total = protocol["ppo_steps_per_run"]
        interval = total//(protocol["evals_per_run"]-1)
        assert [int(r["step"]) for r in rows] == list(range(0, total+1, interval))
        summary = curve_summary(rows)
        for key in summary:
            assert record[key] == summary[key]
        configs.setdefault(meta["shape"], set()).add((run/"env_config.json").read_bytes())
        models.setdefault(meta["shape"], set()).add(meta["model_sha256"])
        if meta["arm"] == "residual":
            assert meta["reference_sha256"] == protocol["reference_sha256"]
        if meta["arm"] == "bc":
            assert meta["dataset_sha256"] == dataset_sha
            assert meta["bc_steps"] == protocol["bc_updates"]
            assert meta["demonstration_loss_weight"] == protocol["bc_demo_regularizer_weight"]
            assert meta["dataset_provenance"]["reference_sha256"] == protocol["reference_sha256"]
        else:
            assert meta["bc_steps"] == 0 and meta["demonstration_loss_weight"] == 0
        checkpoints[(meta["shape"], meta["arm"], meta["seed"])] = (run, summary)
    assert all(len(values) == 1 for values in configs.values())
    assert all(len(values) == 1 for values in models.values())
    assert len(runtime_versions) == 1
    results = json.loads((args.evaluation/"results.json").read_text())
    assert json.loads((args.evaluation/"completed.json").read_text())["runs"] == len(expected)*2
    actual = {(r["task"], r["arm"], r["seed"], r["kind"]) for r in results}
    assert actual == {(*item, kind) for item in expected for kind in ("selected", "terminal")}
    assert len(results) == len(actual)
    low_seed, high_seed = protocol["heldout_seed_range_inclusive"]
    for result in results:
        meta, _ = audit_run(Path(result["evaluation"]), traces)
        assert (meta["jax_version"], meta["mujoco_version"], meta["backend"]) in runtime_versions
        run, summary = checkpoints[(result["task"], result["arm"], result["seed"])]
        step = summary["best_step"] if result["kind"] == "selected" else summary["actual_steps"]
        assert Path(meta["checkpoint"]).resolve() == (run/f"step_{step:09d}/params.pkl").resolve()
        assert meta["eval_start"] == low_seed and meta["eval_count"] == high_seed-low_seed+1
        assert result["seeds"] == list(range(low_seed, high_seed+1))
        assert meta["model_sha256"] in models[result["task"]]
        assert (Path(result["evaluation"])/"env_config.json").read_bytes() in configs[result["task"]]
        assert meta["bound"] == protocol["residual_bound_rad"]
    baseline_shapes = []
    for run in args.baseline_runs:
        meta, _ = audit_run(run, traces)
        assert (meta["jax_version"], meta["mujoco_version"], meta["backend"]) in runtime_versions
        assert meta["zero"] and meta["arm"] == "residual"
        assert meta["eval_start"] == low_seed and meta["eval_count"] == high_seed-low_seed+1
        assert meta["model_sha256"] in models[meta["shape"]]
        assert (run/"env_config.json").read_bytes() in configs[meta["shape"]]
        assert meta["bound"] == protocol["residual_bound_rad"]
        baseline_shapes.append(meta["shape"])
    assert sorted(baseline_shapes) == sorted(protocol["tasks"])
    repeated_arms = set()
    for run in args.repeat_runs:
        meta, _ = audit_run(run, traces)
        assert (meta["jax_version"], meta["mujoco_version"], meta["backend"]) in runtime_versions
        assert meta["shape"] == "cube" and meta["seed"] == 0
        source, summary = checkpoints[("cube", meta["arm"], 0)]
        assert Path(meta["checkpoint"]).resolve() == (source/f"step_{summary['best_step']:09d}/params.pkl").resolve()
        assert meta["eval_start"] == low_seed and meta["eval_count"] == high_seed-low_seed+1
        assert meta["model_sha256"] in models["cube"]
        assert (run/"env_config.json").read_bytes() in configs["cube"]
        assert meta["bound"] == protocol["residual_bound_rad"]
        repeated_arms.add(meta["arm"])
    assert repeated_arms == set(protocol["methods"])
    if args.candidate_repeat_run:
        run = args.candidate_repeat_run
        meta, _ = audit_run(run, traces)
        assert (meta["shape"], meta["arm"], meta["seed"]) == ("cuboid", "residual", 1)
        source, summary = checkpoints[("cuboid", "residual", 1)]
        assert Path(meta["checkpoint"]).resolve() == (source/f"step_{summary['actual_steps']:09d}/params.pkl").resolve()
        assert meta["eval_start"] == low_seed and meta["eval_count"] == high_seed-low_seed+1
        assert meta["model_sha256"] in models["cuboid"]
        assert (run/"env_config.json").read_bytes() in configs["cuboid"]
        assert meta["bound"] == protocol["residual_bound_rad"]
        assert (meta["jax_version"], meta["mujoco_version"], meta["backend"]) in runtime_versions
    report = dict(verified=True, training_runs=len(expected), heldout_runs=len(results),
                  prior_only_runs=len(args.baseline_runs), repeat_runs=len(args.repeat_runs),
                  posthoc_candidate_repeat_runs=int(args.candidate_repeat_run is not None),
                  runtime_versions=sorted(runtime_versions), bc_dataset_sha256=dataset_sha,
                  archived_source_copies_verified=True,
                  hashed_files=traces, scope="all confirmatory/held-out/prior-only/repeat control and grasp traces; not hardware validation")
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "hashed_files"}, indent=2))


if __name__ == "__main__":
    main()
