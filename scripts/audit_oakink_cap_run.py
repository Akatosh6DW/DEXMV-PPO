"""Recompute cap success and control composition from saved physical traces."""

import csv
import json
from pathlib import Path
import pickle

import jax
import mujoco
import numpy as np

from oakink_cap_env import OakInkCap
from prepare_hic_orca_object import digest


def score_trace(env, trace):
    active, alive = trace["active"].astype(bool), trace["alive"].astype(bool)
    np.testing.assert_array_equal(active, np.c_[np.ones(len(active), bool), alive[:, :-1]])
    assert active.shape[1] == 400 and not np.any(alive & ~active)
    released = ~trace["thread_active"].astype(bool)
    assert np.all(np.diff(released.astype(int), axis=1) >= 0)
    np.testing.assert_array_equal(trace["released"], released)
    paired = trace["cap_fingers"][..., 0] & np.any(trace["cap_fingers"][..., 1:], axis=-1)
    np.testing.assert_array_equal(trace["paired"], paired)
    angle = trace["qpos"][..., env.turn]
    np.testing.assert_allclose(trace["degrees"], angle * 180 / np.pi, atol=1e-4)
    displacement = (trace["cap_center"] - np.asarray(env.initial_cap_center)) @ np.asarray(env.bottle_rotation)
    np.testing.assert_allclose(trace["lift"], displacement[..., 2], atol=1e-7)
    bad = np.maximum.accumulate((trace["forbidden"] > 0) | (trace["hand_penetration"] > .005), axis=1)
    bad |= trace["reset_penetration"][:, :1] > .003
    valid = (released & (displacement[..., 2] >= .018) & paired & ~bad
             & ~trace["bottle_contact"].astype(bool) & ~trace["floor_contact"].astype(bool))
    np.testing.assert_array_equal(trace["valid"], valid)
    counts, history = np.zeros(len(active), int), np.zeros(active.shape)
    for t in range(400):
        counts = np.where(valid[:, t], counts + 1, 0)
        history[:, t] = counts * .05
    np.testing.assert_allclose(trace["hold_sec"], history, atol=2e-6)
    final = valid[:, -1] & (counts >= 20) & alive[:, -1]
    np.testing.assert_array_equal((trace["success"][:, -1] > 0) & alive[:, -1], final)
    return dict(success=final, released=np.any(released & alive, axis=1),
                best_hold=np.max(np.where(alive, history, 0), axis=1))


def audit_trace(env, trace):
    if not all(np.isfinite(v).all() for v in trace.values()):
        raise ValueError("Nonfinite cap evaluation")
    outcome = score_trace(env, trace)
    active = trace["active"].astype(bool)
    low, high, initial = np.asarray(env.low), np.asarray(env.high), np.asarray(env.initial)
    ctrl = trace["ctrl"]
    assert (ctrl >= low - 1e-6).all() and (ctrl <= high + 1e-6).all()
    if env.method == "residual":
        base = np.broadcast_to(np.asarray(env.reference_ctrl), ctrl.shape)
        np.testing.assert_allclose(trace["video_base"], base, atol=1e-6)
        delta = np.clip(trace["action"], -1, 1) * env.bound
        np.testing.assert_allclose(trace["requested_residual"], delta, atol=1e-6)
        np.testing.assert_allclose(trace["effective_residual"], ctrl - base, atol=1e-6)
        assert np.abs(trace["effective_residual"]).max() <= env.bound + 1e-5
        desired = np.clip(base + delta, low, high)
    else:
        np.testing.assert_array_equal(trace["obs"][..., -27:-10], 0)
        np.testing.assert_array_equal(trace["video_base"], 0)
        np.testing.assert_array_equal(trace["requested_residual"], 0)
        desired = low + .5 * (np.clip(trace["action"], -1, 1) + 1) * (high - low)
    np.testing.assert_allclose(trace["desired_ctrl"], desired, atol=1e-6)
    previous = np.concatenate([np.broadcast_to(initial, (len(ctrl), 1, 17)), ctrl[:, :-1]], axis=1)
    np.testing.assert_allclose(ctrl, np.clip(desired, previous - .1, previous + .1), atol=1e-6)
    if not env.tactile:
        np.testing.assert_array_equal(trace["obs"][..., -10:], 0)
    geometry = mujoco.MjData(env.mj_model)
    from oakink_cap_dynamics import apply_thread_contact_mode, contact_report
    for i in range(len(ctrl)):
        mujoco.mj_resetData(env.mj_model, geometry)
        geometry.qpos[:] = trace["reset_qpos"][i, 0]
        geometry.ctrl[:] = initial
        apply_thread_contact_mode(env.mj_model, geometry)
        mujoco.mj_forward(env.mj_model, geometry)
        depths = contact_report(env.mj_model, geometry)
        expected = max(depths["hand_cap_depth_m"], depths["hand_bottle_depth_m"])
        np.testing.assert_allclose(trace["reset_penetration"][i], expected, atol=2e-6)
        np.testing.assert_array_equal(trace["reset_qpos"][i], np.broadcast_to(geometry.qpos.astype(np.float32), trace["reset_qpos"][i].shape))
    return dict(success=float(outcome["success"].mean()), release_rate=float(outcome["released"].mean()),
                mean_best_hold_s=float(outcome["best_hold"].mean()),
                max_penetration_mm=float(trace["hand_penetration"][active].max() * 1000),
                episodes={k: v.tolist() for k, v in outcome.items()})


def audit_run(run):
    run = Path(run)
    cfg = json.loads((run / "run_config.json").read_text())
    if digest(run / "model.mjb") != cfg["binary_model_sha256"]:
        raise ValueError("Saved cap model changed")
    if not cfg["development"]:
        from oakink_cap_protocol import check_protocol
        protocol = check_protocol(cfg["protocol"])
        if digest(cfg["protocol"]) != cfg["protocol_sha256"]:
            raise ValueError("Training protocol changed")
        recipe = next(r for r in protocol["recipes"] if r["label"] == run.name)
        assert all(cfg[key] == recipe[key] for key in ("seed", "method", "steps"))
    for name, expected in cfg["sources"].items():
        if digest(run / name) != expected:
            raise ValueError("Training source snapshot changed")
    env = OakInkCap(cfg["model"], cfg["reference"], "residual" if cfg["method"] == "residual" else "pure", cfg["tactile"] == "on")
    if digest(env.xml_path) != cfg["model_sha256"] or digest(cfg["reference"]) != cfg["reference_sha256"]:
        raise ValueError("Run model/reference changed")
    if cfg["method"] == "guided":
        if digest(cfg["bc_checkpoint"]) != cfg["bc_checkpoint_sha256"]:
            raise ValueError("Paired BC checkpoint changed")
        with Path(cfg["bc_checkpoint"]).open("rb") as f:
            expected = pickle.load(f)
        with (run / "initial_params.pkl").open("rb") as f:
            actual = pickle.load(f)
        assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
        for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
            np.testing.assert_array_equal(a, b)
    with (run / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert int(rows[0]["step"]) == 0 and int(rows[-1]["step"]) == cfg["steps"]
    assert len(rows) == (cfg["evals"] if cfg["steps"] else 1)
    audits = []
    for row in rows:
        folder = run / f"step_{int(row['step']):09d}"
        with np.load(folder / "trajectories.npz") as packet:
            result = audit_trace(env, dict(packet))
        for key in ("success", "release_rate", "mean_best_hold_s"):
            np.testing.assert_allclose(float(row[key]), result[key], atol=1e-6)
        audits.append(dict(step=int(row["step"]), trace_sha256=digest(folder / "trajectories.npz"),
                           params_sha256=digest(folder / "params.pkl"), **result))
    return dict(passed=True, source_sha256=digest(Path(__file__)), checkpoints=audits,
                scope="Final-state success, initial geometry, executed control, source identity and paired BC weights")
