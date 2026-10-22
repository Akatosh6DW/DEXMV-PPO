"""Check held-out provenance, actual reset/physics, and scores before pooling."""

import json
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

from orca_formal_protocol import scenarios
from prepare_hic_orca_object import digest

ROOT = Path(__file__).resolve().parents[1]


def check_report_identity(report, job, suite, development=False):
    if Path(report["run"]).resolve() != (suite / job["run"]).resolve():
        raise ValueError("Evaluation run identity differs from declared job")
    if report["scenario"] != job["scenario"] or report["overrides"] != scenarios()[job["scenario"]]:
        raise ValueError("Evaluation scenario differs from declared job")
    if bool(report["zero_correction_baseline"]) != bool(job["zero"]):
        raise ValueError("Wrong zero-correction baseline flag")
    if job["zero"] and report["reference_mode"] != job["zero"]:
        raise ValueError("Wrong zero-correction reference")
    expected_seeds = [19000, 19001] if development else list(range(25000, 25064))
    if report["seeds"] != expected_seeds or report["development_smoke"] != development:
        raise ValueError("Incorrect independent reset set")


def independent_success(trace):
    active = trace["active"].astype(bool)
    alive = trace["alive"].astype(bool)
    if active.ndim != 2 or active.shape[1] != 800:
        raise ValueError("Expected complete 40-second episode arrays")
    np.testing.assert_array_equal(active, np.c_[np.ones(len(active), bool), alive[:, :-1]])
    valid = (active & alive & (trace["released"] > 0) & (trace["thumb"] > 0)
             & (trace["contacts"] >= 2) & (trace["forbidden"] == 0)
             & (trace["cube_z"] > trace["spawn_z"] - .01))
    count, longest = np.zeros(len(active), int), np.zeros(len(active), int)
    history = np.zeros(active.shape)
    for step in range(800):
        count = np.where(valid[:, step], count + 1, 0)
        longest = np.maximum(longest, count)
        history[:, step] = count * .05
    np.testing.assert_allclose(trace["hold"], history, atol=1e-6)
    valid_initial = trace["initial_penetration_m"][:, 0] <= .003
    raw = (longest >= 600) & valid[:, -1]
    strict = raw & valid_initial
    return dict(raw=raw, strict=strict, hold=longest * .05, valid_initial=valid_initial,
                success15=(longest >= 300) & valid[:, -1] & valid_initial)


def audit_reset_geometry(model, trace, initial):
    cube = model.geom("cube").id
    support = model.geom("cube_support").id
    if model.geom_type[cube] == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.asarray(list(product(*zip(-model.geom_size[cube], model.geom_size[cube]))))
    elif model.geom_type[cube] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[cube]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count]
    else:
        raise ValueError("Unsupported object geometry in reset audit")
    data = mujoco.MjData(model)
    for i in range(len(trace["reset_qpos"])):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = trace["reset_qpos"][i, 0]
        data.mocap_pos[:] = trace["reset_mocap"][i, 0]
        data.ctrl[:] = initial
        mujoco.mj_forward(model, data)
        world = vertices @ data.geom_xmat[cube].reshape(3, 3).T + data.geom_xpos[cube]
        support_top = data.geom_xpos[support, 2] + model.geom_size[support, 2]
        np.testing.assert_allclose(support_top, world[:, 2].min(), atol=2e-7)
        np.testing.assert_allclose(data.geom_xpos[support, :2], data.xpos[model.geom_bodyid[cube], :2], atol=2e-7)
        depth = max([0.] + [-float(c.dist) for c in data.contact if cube in (c.geom1, c.geom2)])
        np.testing.assert_allclose(trace["initial_penetration_m"][i], depth, atol=2e-7)
        np.testing.assert_array_equal(trace["reset_mocap"][i],
                                      np.broadcast_to(data.mocap_pos, trace["reset_mocap"][i].shape))


def audit(directory, job, suite, development=False):
    report = json.loads((directory / "report.json").read_text())
    check_report_identity(report, job, suite, development)
    run = suite / job["run"]
    cfg = json.loads((run / "run_config.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    nominal = mujoco.MjModel.from_xml_path(str(ROOT / cfg["model"]))
    for filename, key in (("trajectories.npz", "trace_sha256"), ("model.mjb", "actual_model_sha256"),
                          ("model.xml", "model_xml_sha256")):
        if digest(directory / filename) != report[key]:
            raise ValueError("Evaluation artifact hash changed: " + filename)
    if digest(run / f"step_{job['step']:09d}/params.pkl") != report["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    if digest(ROOT / cfg["reference"]) != report["reference_sha256"]:
        raise ValueError("Reference changed")
    assert report["train_seed"] == cfg["seed"] and report["method"] == cfg["method"]
    assert report["tactile"] == cfg["tactile"]
    assert report["reference_mode"] == (job["zero"] or cfg.get("reference_mode", "video"))
    for name, value in report["source_sha256"].items():
        if digest(ROOT / "scripts" / name) != value:
            raise ValueError("Evaluator source changed: " + name)
    scenario = scenarios()[job["scenario"]]
    cube = model.body("cube").id
    expected_mass, expected_inertia = nominal.body_mass.copy(), nominal.body_inertia.copy()
    expected_mass[cube] *= scenario.get("mass_scale", 1.)
    expected_inertia[cube] *= scenario.get("mass_scale", 1.)
    np.testing.assert_allclose(model.body_mass, expected_mass, atol=1e-12)
    np.testing.assert_allclose(model.body_inertia, expected_inertia, atol=1e-12)
    friction = nominal.geom_friction.copy()
    physical = (nominal.geom_contype != 0) | (nominal.geom_conaffinity != 0)
    friction[physical] *= scenario.get("friction_scale", 1.)
    np.testing.assert_allclose(model.geom_friction, friction, atol=1e-12)
    with np.load(directory / "trajectories.npz") as packet:
        trace = dict(packet)
    assert len(trace["active"]) == len(report["seeds"])
    assert all(np.isfinite(value).all() for value in trace.values())
    outcome = independent_success(trace)
    metrics = report["metrics"]
    for key, value in (("success30", outcome["strict"].mean()), ("raw_success30", outcome["raw"].mean()),
                       ("success15", outcome["success15"].mean()),
                       ("strict_mean_hold", np.where(outcome["valid_initial"], outcome["hold"], 0).mean())):
        np.testing.assert_allclose(metrics[key], value, atol=1e-8, err_msg=key)
    for key, value in (("success30", "strict"), ("raw_success30", "raw"),
                       ("valid_initial", "valid_initial"), ("hold_sec", "hold")):
        np.testing.assert_allclose(report["episodes"][key], outcome[value], atol=1e-8)
    np.testing.assert_array_equal(trace["obs"][..., 94], 0.)
    np.testing.assert_array_equal(trace["obs"][..., -10:], trace["touch_obs"])
    if cfg["tactile"] == "off":
        np.testing.assert_array_equal(trace["touch_obs"], 0.)
    active = trace["active"].astype(bool)
    time = (np.arange(800) + 1) * .05
    force = np.zeros((800, 3))
    force[240:244, 0], force[480:484, 0] = scenario.get("push_force_n", 0.), -scenario.get("push_force_n", 0.)
    expected_force = np.broadcast_to(force, trace["external_force_n"].shape)
    np.testing.assert_array_equal(trace["external_force_n"][active], expected_force[active])
    np.testing.assert_array_equal(trace["released"][active], np.broadcast_to(time >= 3, active.shape)[active])
    initial = np.asarray(json.loads((run / "env_config.json").read_text())["initial_ctrl"])
    audit_reset_geometry(model, trace, initial)
    hand = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    cube_qid = model.joint("cube_freejoint").qposadr[0]
    low, high = model.actuator_ctrlrange.T
    spawn = nominal.body("cube").pos.copy()
    orientation = nominal.body("cube").quat.copy()
    for i, seed in enumerate(report["seeds"]):
        rng = np.random.default_rng(seed)
        position = spawn + rng.uniform(-1, 1, 3) * [scenario.get("xy_jitter_m", .002)] * np.array([1., 1., 0.])
        joints = np.clip(initial + rng.normal(0, .002, 17), low, high)
        quaternion = orientation.copy()
        if scenario.get("yaw_deg"):
            angle_rng = np.random.default_rng(np.random.SeedSequence([seed, 55903]))
            angle = np.deg2rad(angle_rng.uniform(-scenario["yaw_deg"], scenario["yaw_deg"]))
            mujoco.mju_mulQuat(quaternion, np.array([np.cos(angle / 2), 0., 0., np.sin(angle / 2)]), orientation)
        reset = trace["reset_qpos"][i, 0]
        np.testing.assert_allclose(reset[hand], joints, atol=2e-7)
        np.testing.assert_allclose(reset[cube_qid:cube_qid + 3], position, atol=2e-7)
        np.testing.assert_allclose(reset[cube_qid + 3:cube_qid + 7], quaternion, atol=2e-7)
        np.testing.assert_array_equal(trace["reset_qpos"][i], np.broadcast_to(reset, trace["reset_qpos"][i].shape))
    ctrl = trace["ctrl"]
    assert np.all(ctrl >= low - 1e-6) and np.all(ctrl <= high + 1e-6)
    assert np.max(abs(ctrl[:, 0] - initial)) <= .10001
    assert np.max(abs(np.diff(ctrl, axis=1))) <= .10001
    if cfg["method"] == "residual":
        from orca_matched_env import reference_variant
        from orca_residual_controls import load_reference
        base, _ = load_reference(ROOT / cfg["reference"], low, high, initial, .05, 800, .1)
        base = reference_variant(base, initial, report["reference_mode"])
        np.testing.assert_allclose(trace["video_base"][active], np.broadcast_to(base, ctrl.shape)[active], atol=1e-6)
        delta = np.clip(trace["action"], -1, 1) * cfg["residual_bound"]
        np.testing.assert_allclose(trace["requested_residual"], delta, atol=1e-6)
        desired = np.clip(trace["video_base"] + delta, low, high)
        if job["zero"]:
            np.testing.assert_array_equal(delta, 0.)
    else:
        np.testing.assert_array_equal(trace["video_base"], 0.)
        np.testing.assert_array_equal(trace["requested_residual"], 0.)
        np.testing.assert_array_equal(trace["obs"][..., 95:112], 0.)
        desired = low + (np.clip(trace["action"], -1, 1) + 1) * .5 * (high - low)
    np.testing.assert_allclose(trace["desired_ctrl"], desired, atol=1e-6)
    previous = np.concatenate([np.broadcast_to(initial, (len(ctrl), 1, 17)), ctrl[:, :-1]], axis=1)
    np.testing.assert_allclose(ctrl[active], np.clip(desired, previous - .1, previous + .1)[active], atol=1e-6)
    return dict(passed=True, report_sha256=digest(directory / "report.json"),
                source_sha256=digest(Path(__file__)), episodes=len(ctrl),
                scope="Independent hold scoring, declared job identity, native model physics, resets, control composition and actor masking")
