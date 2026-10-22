"""Recompute formal-run scores and check provenance and executed controls."""

import csv
import json
from pathlib import Path
import pickle

import jax
import mujoco
import numpy as np

from orca_matched_env import reference_variant
from orca_residual_controls import load_reference
from orca_residual_metrics import summarize
from prepare_hic_orca_object import digest

ROOT = Path(__file__).resolve().parents[1]


def audit_run(run, recipe, protocol):
    config = json.loads((run / "run_config.json").read_text())
    environment = json.loads((run / "env_config.json").read_text())
    for key in ("method", "tactile", "seed", "reference_mode", "steps"):
        if config[key] != recipe[key]:
            raise ValueError("Formal recipe mismatch: " + key)
    assert config["init_std"] == recipe["std"] and config["residual_bound"] == recipe["bound"]
    assert config["learning_rate"] == .0003 and config["eval_start"] == 11000 and config["eval_count"] == 16
    assert config["physics_backend"] == "native_mujoco_callback" and config["backend"] == "gpu"
    for key in ("model", "reference", "gate", "tactile_gate"):
        path = ROOT / config[key]
        assert digest(path) == config[key + "_sha256"] == protocol["inputs"][str(path.resolve())]
    for name, expected in config["source_sha256"].items():
        assert digest(run / "sources" / name) == expected == protocol["sources"][name]
    if recipe["method"] in ("bc", "guided"):
        assert config["dataset_sha256"] == protocol["dataset_sha256"]
    model = mujoco.MjModel.from_xml_path(str(ROOT / config["model"]))
    low, high = model.actuator_ctrlrange.T
    base, _ = load_reference(ROOT / config["reference"], low, high, environment["initial_ctrl"], .05, 800, .1)
    base = reference_variant(base, environment["initial_ctrl"], recipe["reference_mode"])
    with (run / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected_steps = [0] if recipe["method"] == "bc" else np.linspace(0, recipe["steps"], 9, dtype=int).tolist()
    assert [int(r["step"]) for r in rows] == expected_steps
    traces = []
    for row in rows:
        directory = run / f"step_{int(row['step']):09d}"
        with np.load(directory / "trajectories.npz") as packet:
            trace = dict(packet)
        assert all(np.isfinite(v).all() for v in trace.values())
        arm = "residual" if recipe["method"] == "residual" else "pure"
        scored, holds, success = summarize(trace, arm, recipe["bound"], low, high)
        for key, value in scored.items():
            np.testing.assert_allclose(value, float(row[key]), atol=1e-5)
        episodes = json.loads((directory / "episodes.json").read_text())
        assert episodes["seeds"] == list(range(11000, 11016))
        np.testing.assert_allclose(holds, episodes["hold_sec"], atol=1e-5)
        np.testing.assert_array_equal(success, episodes["success30"])
        np.testing.assert_array_equal(trace["active"], np.c_[np.ones(len(holds), bool), trace["alive"][:, :-1]])
        np.testing.assert_array_equal(trace["obs"][..., 94], 0.)
        np.testing.assert_array_equal(trace["obs"][..., -10:], trace["touch_obs"])
        if recipe["tactile"] == "off":
            np.testing.assert_array_equal(trace["touch_obs"], 0.)
        if arm == "residual":
            np.testing.assert_allclose(trace["video_base"], np.broadcast_to(base, trace["video_base"].shape), atol=1e-6)
        else:
            np.testing.assert_array_equal(trace["video_base"], 0.)
            np.testing.assert_array_equal(trace["obs"][..., 95:112], 0.)
        assert np.abs(trace["ctrl"][:, 0] - environment["initial_ctrl"]).max() <= .10001
        params = pickle.loads((directory / "params.pkl").read_bytes())
        assert all(np.isfinite(v).all() for v in jax.tree.leaves(params))
        if int(row["step"]) == 0 and recipe["bc"]:
            bc_path = run.parent / recipe["bc"] / "bc_params.pkl"
            assert digest(bc_path) == config["bc_checkpoint_sha256"]
            previous = pickle.loads(bc_path.read_bytes())
            assert jax.tree.structure(previous) == jax.tree.structure(params)
            for a, b in zip(jax.tree.leaves(previous), jax.tree.leaves(params)):
                np.testing.assert_array_equal(a, b)
        traces.append(dict(step=int(row["step"]), sha256=digest(directory / "trajectories.npz")))
    return dict(passed=True, label=recipe["label"], traces=traces,
                scope="Input provenance, paired BC initialization, executed controls and independent episode scores")
