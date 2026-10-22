"""Re-score saved physical trajectories and verify immutable experiment inputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import jax
import mujoco
import numpy as np

from orca_residual_metrics import summarize

ROOT = Path(__file__).resolve().parents[1]


def audit_trace(path, metrics, episodes, arm, low, high):
    with np.load(path, allow_pickle=False) as z:
        trace = {name: z[name] for name in z.files}
    for name, values in trace.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {path}:{name}")
    row, holds, success = summarize(trace, "residual" if arm == "residual" else "pure", .08, low, high)
    for name, value in row.items():
        np.testing.assert_allclose(value, float(metrics[name]), atol=1e-5)
    np.testing.assert_allclose(holds, episodes["hold_sec"], atol=1e-5)
    np.testing.assert_array_equal(success, episodes["success30"])
    active = np.concatenate([np.ones((len(holds), 1), bool), trace["alive"][:, :-1]], axis=1)
    np.testing.assert_array_equal(trace["active"], active)
    np.testing.assert_array_equal(trace["reward"][~active], 0)
    np.testing.assert_array_less(np.abs(trace["ctrl"][:, 0] - np.array([-.25] + [0.] * 16)), .10001)
    correction = trace["requested_residual"][active]
    extra = dict(final_hold30=float(np.mean(trace["hold"][:, -1] >= 30)),
                 active_residual_rms_rad=float(np.sqrt(np.mean(correction ** 2))),
                 active_residual_saturation=float(np.mean(np.abs(correction) >= .0792)),
                 active_action_rms=float(np.sqrt(np.mean(trace["action"][active] ** 2))))
    for name, value in extra.items():
        np.testing.assert_allclose(value, float(metrics[name]), atol=1e-5)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, type=Path)
    p.add_argument("--heldout", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/orca_residual/models/cube.xml"))
    low, high = model.actuator_ctrlrange.T
    proto = json.loads((args.suite / "protocol.json").read_text())
    if not (args.suite / "completed.json").exists():
        raise ValueError("Incomplete training")
    rows = []
    configs = []
    expected_cases = set()
    initial_params = {}
    env_configs = []
    for seed in proto["seeds"]:
        for reward in proto["rewards"]:
            for arm in proto["arms"]:
                std = proto["chosen"][arm]["std"]
                run = args.suite / f"{reward}_{arm}_std{std}_seed{seed}"
                config = json.loads((run / "run_config.json").read_text())
                assert (config["seed"], config["arm"], config["reward"], config["init_std"]) == (seed, arm, reward, std)
                assert config["demo_loss_weight"] == 0
                assert config["bc_updates"] == (2000 if arm == "bc" else 0)
                for name, expected in config["source_sha256"].items():
                    for source in (run / "sources" / name, ROOT / "scripts" / name):
                        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, str(source)
                configs.append(config)
                env_configs.append(json.loads((run / "env_config.json").read_text()))
                with (run / "step_000000000" / "params.pkl").open("rb") as f:
                    initial = pickle.load(f)
                key = (seed, arm)
                if key in initial_params:
                    assert jax.tree.structure(initial) == jax.tree.structure(initial_params[key])
                    for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(initial_params[key])):
                        np.testing.assert_array_equal(a, b)
                else:
                    initial_params[key] = initial
                for label in ("initial", "terminal"):
                    expected_cases.add((run.name, label))
                for line in (run / "ppo_metrics.jsonl").read_text().splitlines():
                    assert np.isfinite(list(json.loads(line).values())).all()
                with (run / "metrics.csv").open() as f:
                    metrics = list(csv.DictReader(f))
                np.testing.assert_array_equal([int(m["step"]) for m in metrics],
                                              np.linspace(0, proto["steps"], proto["evals"], dtype=int))
                for metric in metrics:
                    checkpoint = run / f"step_{int(metric['step']):09d}"
                    with (checkpoint / "params.pkl").open("rb") as f:
                        params = pickle.load(f)
                    assert all(np.isfinite(value).all() for value in jax.tree.leaves(params))
                    episodes = json.loads((checkpoint / "episodes.json").read_text())
                    assert episodes["seeds"] == list(range(proto["eval_start"], proto["eval_start"] + proto["eval_count"]))
                    path = checkpoint / "trajectories.npz"
                    digest = audit_trace(path, metric, episodes, arm, low, high)
                    rows.append(dict(path=str(path), sha256=digest))
    for name in ("source_sha256", "dataset_sha256", "reference_sha256", "model_sha256",
                 "discounting", "entropy_cost", "learning_rate", "num_envs", "batch_size",
                 "num_minibatches", "unroll_length", "num_updates_per_batch"):
        assert all(c[name] == configs[0][name] for c in configs), name
    assert all(c == env_configs[0] for c in env_configs)
    for name, path in (("model", ROOT / "artifacts/orca_residual/models/cube.xml"),
                       ("reference", ROOT / "artifacts/orca30/dexmv_slow_acquisition_o21/reference.npz"),
                       ("dataset", ROOT / "artifacts/orca_residual/shared_cube_demos/trajectories.npz")):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == configs[0][name + "_sha256"]
    report = json.loads((args.heldout / "report.json").read_text())
    assert report["complete"] and len(report["records"]) == len(configs) * 2
    assert {(r["run"], r["label"]) for r in report["records"]} == expected_cases
    heldout_protocol = json.loads((args.heldout / "protocol.json").read_text())
    assert heldout_protocol["protocol_sha256"] == hashlib.sha256((args.suite / "protocol.json").read_bytes()).hexdigest()
    for record in report["records"]:
        checkpoint = Path(record["checkpoint"])
        step = 0 if record["label"] == "initial" else proto["steps"]
        assert record["checkpoint_step"] == step
        assert checkpoint.resolve() == (args.suite / record["run"] / f"step_{step:09d}" / "params.pkl").resolve()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record["checkpoint_sha256"]
        directory = args.heldout / f"{record['run']}_{record['label']}"
        episodes = json.loads((directory / "episodes.json").read_text())
        assert episodes["seeds"] == list(range(6000, 6032))
        path = directory / "trajectories.npz"
        rows.append(dict(path=str(path), sha256=audit_trace(path, record, episodes, record["arm"], low, high)))
    args.output.write_text(json.dumps(dict(complete=True, training_runs=len(configs),
                                         heldout_evaluations=len(report["records"]), traces=rows), indent=2))
    print(json.dumps(dict(training_runs=len(configs), heldout_evaluations=len(report["records"]),
                         traces=len(rows), status="PASS")))


if __name__ == "__main__":
    main()
