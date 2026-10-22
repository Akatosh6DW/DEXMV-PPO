"""Independently verify matched-study inputs, controls, scores and BC pairing."""

import argparse
import csv
import json
from pathlib import Path
import pickle

import jax
import mujoco
import numpy as np

from orca_residual_controls import load_reference
from orca_residual_metrics import summarize
from prepare_hic_orca_object import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    args = p.parse_args()
    suite = args.suite.resolve()
    protocol = json.loads((suite / "protocol.json").read_text())
    result = json.loads((suite / "results.json").read_text())
    expected = {(m, t) for m in ("bc", "ppo", "guided", "residual") for t in ("off", "on")}
    assert result["complete"] and len(result["records"]) == 8
    assert {(r["method"], r["tactile"]) for r in result["records"]} == expected
    manifest = json.loads((suite / "teacher/dataset_manifest.json").read_text())
    assert manifest["dataset_sha256"] == digest(suite / "teacher/trajectories.npz")
    assert manifest["source_video_count"] == 1 and manifest["source_frames"] == [30, 95]
    configs, traces, initial_params = [], [], {}
    for record in result["records"]:
        method, tactile = record["method"], record["tactile"]
        run = suite / record["run"]
        config = json.loads((run / "run_config.json").read_text())
        env_config = json.loads((run / "env_config.json").read_text())
        configs.append(config)
        assert (config["method"], config["tactile"], config["seed"]) == (method, tactile, protocol["seed"])
        assert config["physics_backend"] == manifest["physics_backend"] == protocol["physics_backend"]
        for key in ("model", "reference"):
            assert digest(ROOT / config[key]) == config[key + "_sha256"] == protocol[key + "_sha256"]
        for name, expected_hash in config["source_sha256"].items():
            assert digest(run / "sources" / name) == expected_hash
        if method in ("bc", "guided"):
            assert config["dataset_sha256"] == manifest["dataset_sha256"]
        model = mujoco.MjModel.from_xml_path(str(ROOT / config["model"]))
        low, high = model.actuator_ctrlrange.T
        reference, _ = load_reference(ROOT / config["reference"], low, high, env_config["initial_ctrl"], .05, 800, .1)
        with (run / "metrics.csv").open() as stream:
            metrics = list(csv.DictReader(stream))
        expected_steps = [0] if method == "bc" else np.linspace(0, protocol["ppo_steps"], 5, dtype=int).tolist()
        assert [int(row["step"]) for row in metrics] == expected_steps
        for row in metrics:
            checkpoint = run / f"step_{int(row['step']):09d}"
            with np.load(checkpoint / "trajectories.npz") as packet:
                trace = dict(packet)
            assert all(np.isfinite(v).all() for v in trace.values())
            scored, holds, success = summarize(trace, "residual" if method == "residual" else "pure", .08, low, high)
            for key, value in scored.items():
                np.testing.assert_allclose(value, float(row[key]), atol=1e-5)
            episodes = json.loads((checkpoint / "episodes.json").read_text())
            assert episodes["seeds"] == protocol["evaluation_seeds"]
            np.testing.assert_allclose(holds, episodes["hold_sec"], atol=1e-5)
            np.testing.assert_array_equal(success, episodes["success30"])
            final_valid = trace["alive"][:, -1] & (trace["hold"][:, -1] > 0)
            np.testing.assert_allclose(np.mean((holds >= 15) & final_valid), float(row["success15"]))
            np.testing.assert_allclose(np.mean(holds >= 15), float(row["success15_any"]))
            np.testing.assert_array_equal(trace["active"], np.c_[np.ones(len(holds), bool), trace["alive"][:, :-1]])
            np.testing.assert_array_equal(trace["reward"][~trace["active"]], 0.)
            np.testing.assert_array_equal(trace["obs"][..., 94], 0.)
            np.testing.assert_array_equal(trace["obs"][..., -10:], trace["touch_obs"])
            if tactile == "off":
                np.testing.assert_array_equal(trace["touch_obs"], 0.)
            if method == "residual":
                np.testing.assert_allclose(trace["video_base"], np.broadcast_to(reference, trace["video_base"].shape), atol=1e-6)
            else:
                np.testing.assert_array_equal(trace["video_base"], 0.)
                np.testing.assert_array_equal(trace["obs"][..., 95:112], 0.)
            assert np.max(np.abs(trace["ctrl"][:, 0] - env_config["initial_ctrl"])) <= .10001
            params = pickle.loads((checkpoint / "params.pkl").read_bytes())
            assert all(np.isfinite(v).all() for v in jax.tree.leaves(params))
            if int(row["step"]) == 0:
                initial_params[(method, tactile)] = params
            traces.append(dict(path=str(checkpoint / "trajectories.npz"), sha256=digest(checkpoint / "trajectories.npz")))
    for key in ("source_sha256", "weights", "learning_rate", "seed", "eval_start", "eval_count",
                "model_sha256", "reference_sha256", "gate_sha256", "tactile_gate_sha256", "xla_flags"):
        assert all(c[key] == configs[0][key] for c in configs), key
    for tactile in ("off", "on"):
        bc, guided = initial_params[("bc", tactile)], initial_params[("guided", tactile)]
        assert jax.tree.structure(bc) == jax.tree.structure(guided)
        for a, b in zip(jax.tree.leaves(bc), jax.tree.leaves(guided)):
            np.testing.assert_array_equal(a, b)
    report = dict(pass_audit=True, runs=8, traces=traces, note="Single-seed pilot; input/control/metric audit, not hardware validation")
    (suite / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", len(traces), "traces")


if __name__ == "__main__":
    main()
