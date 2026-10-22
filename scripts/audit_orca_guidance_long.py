"""Independently re-score the complete long study and verify declared inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import jax
import mujoco
import numpy as np

from analyze_orca_guidance_recheck import read_runs
from audit_orca_guidance_recheck import audit_trace
from evaluate_orca_guidance_long import condition_config
from orca_guidance_recheck import ROOT, REFERENCE, DATASET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--nominal", type=Path, required=True)
    p.add_argument("--offset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    proto, runs = read_runs(args.suite)
    assert proto["steps"] == 2621440 and proto["evals"] == 17
    assert proto["seeds"] == [20, 21, 22, 23, 24]
    assert len(runs) == 30
    expected_runs = {f"{reward}_{arm}_std{proto['chosen'][arm]['std']}_seed{seed}"
                     for reward in proto["rewards"] for arm in proto["arms"] for seed in proto["seeds"]}
    assert {r["run"] for r in runs} == expected_runs
    model_path = ROOT / "artifacts/orca_residual/models/cube.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    low, high = model.actuator_ctrlrange.T
    files, configs, env_configs, initials = [], [], [], {}
    for run in runs:
        directory = args.suite / run["run"]
        config = json.loads((directory / "run_config.json").read_text())
        assert (config["steps"], config["evals"], config["eval_start"], config["eval_count"]) == (2621440, 17, 5100, 16)
        assert config["demo_loss_weight"] == 0
        assert config["bc_updates"] == (2000 if run["arm"] == "bc" else 0)
        for name, expected in config["source_sha256"].items():
            assert digest(directory / "sources" / name) == digest(ROOT / "scripts" / name) == expected
        for name, path in (("reference", REFERENCE), ("dataset", DATASET), ("model", model_path)):
            assert digest(path) == config[name+"_sha256"]
        configs.append(config)
        env_configs.append(json.loads((directory / "env_config.json").read_text()))
        for line in (directory / "ppo_metrics.jsonl").read_text().splitlines():
            assert np.isfinite(list(json.loads(line).values())).all()
        np.testing.assert_array_equal([r["step"] for r in run["curve"]], np.arange(17)*163840)
        for metric in run["curve"]:
            checkpoint = directory / f"step_{int(metric['step']):09d}"
            params = pickle.loads((checkpoint / "params.pkl").read_bytes())
            assert all(np.isfinite(v).all() for v in jax.tree.leaves(params))
            if metric["step"] == 0:
                key = (run["arm"], run["seed"])
                if key in initials:
                    assert jax.tree.structure(params) == jax.tree.structure(initials[key])
                    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(initials[key])):
                        np.testing.assert_array_equal(a, b)
                else:
                    initials[key] = params
            episodes = json.loads((checkpoint / "episodes.json").read_text())
            assert episodes["seeds"] == list(range(5100, 5116))
            path = checkpoint / "trajectories.npz"
            files.append(dict(path=str(path), sha256=audit_trace(path, metric, episodes, run["arm"], low, high)))
    for key in ("source_sha256", "learning_rate", "entropy_cost", "discounting", "num_envs", "batch_size",
                "num_minibatches", "num_updates_per_batch", "unroll_length", "backend", "precision"):
        assert all(c[key] == configs[0][key] for c in configs), key
    assert all(e == env_configs[0] for e in env_configs)
    for condition, directory in (("nominal", args.nominal), ("offset", args.offset)):
        cfg = condition_config(condition)
        manifest = json.loads((directory / "protocol.json").read_text())
        assert manifest["protocol_sha256"] == digest(args.suite / "protocol.json")
        assert manifest["evaluator_sha256"] == digest(ROOT / "scripts/evaluate_orca_guidance_long.py")
        for key in cfg:
            assert cfg[key] == manifest[key]
        report = json.loads((directory / "report.json").read_text())
        records = report["records"]
        assert report["complete"]
        expected = {(r, label) for r in expected_runs for label in cfg["checkpoints"]}
        assert len(records) == len(expected) and {(r["run"], r["label"]) for r in records} == expected
        for record in records:
            step = cfg["checkpoints"][record["label"]]
            checkpoint = args.suite / record["run"] / f"step_{step:09d}" / "params.pkl"
            assert Path(record["checkpoint"]).resolve() == checkpoint.resolve()
            assert record["checkpoint_step"] == step and record["checkpoint_sha256"] == digest(checkpoint)
            case = directory / f"{record['run']}_{record['label']}"
            episodes = json.loads((case / "episodes.json").read_text())
            assert episodes["seeds"] == cfg["seeds"]
            path = case / "trajectories.npz"
            files.append(dict(path=str(path), sha256=audit_trace(path, record, episodes, record["arm"], low, high)))
            with np.load(path) as trace:
                offsets = trace["mocap"][:, 0, 0, :2]-np.array([.055, .025])
                assert np.max(np.abs(offsets)) <= cfg["jitter"]+1e-6
                assert np.max(np.abs(offsets)) >= cfg["jitter"]*.5
    result = dict(complete=True, training_runs=30, training_checkpoints=510, heldout_evaluations=150,
                  traces=files, physical_episodes=30*17*16+150*64)
    assert len(files) == 660
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: len(v) if k == "traces" else v for k, v in result.items()}))


if __name__ == "__main__":
    main()
