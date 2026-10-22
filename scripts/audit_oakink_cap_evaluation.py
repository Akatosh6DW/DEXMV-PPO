"""Independent provenance, perturbation, reset and outcome audit for cap tests."""

import json
from pathlib import Path

import jax
import mujoco
import numpy as np

from audit_oakink_cap_run import audit_trace
from oakink_cap_env import OakInkCap
from oakink_cap_protocol import ROOT, scenarios
from prepare_hic_orca_object import digest


def audit(directory, job, suite, development=False):
    report = json.loads((directory / "report.json").read_text())
    run = suite / job["run"]
    if Path(report["run"]).resolve() != run.resolve() or report["baseline"] != job["baseline"] or report["scenario"] != job["scenario"]:
        raise ValueError("Wrong cap evaluation identity")
    if report["seeds"] != ([19020, 19021] if development else list(range(28500, 28564))) or report["development"] != development:
        raise ValueError("Wrong cap test population")
    cfg = json.loads((run / "run_config.json").read_text())
    assert report["method"] == cfg["method"] and report["train_seed"] == cfg["seed"]
    assert report["overrides"] == scenarios()[job["scenario"]]
    assert report["protocol_sha256"] == cfg.get("protocol_sha256")
    for path, key in ((directory / "trajectories.npz", "trace_sha256"), (directory / "model.mjb", "model_sha256"),
                      (Path(cfg["reference"]), "reference_sha256"), (Path(cfg["model"]) / "model.xml", "model_source_sha256")):
        if digest(path) != report[key]:
            raise ValueError("Cap evaluation input/artifact changed: " + str(path))
    if not job["baseline"] and digest(run / f"step_{cfg['steps']:09d}/params.pkl") != report["checkpoint_sha256"]:
        raise ValueError("Cap checkpoint changed")
    for name, expected in report["sources"].items():
        if digest(ROOT / "scripts" / name) != expected:
            raise ValueError("Cap evaluation source changed: " + name)
    method = ("residual" if job["baseline"] == "video" else "pure") if job["baseline"] else (
        "residual" if cfg["method"] == "residual" else "pure")
    env = OakInkCap(cfg["model"], cfg["reference"], method, cfg["tactile"] == "on")
    nominal = env.mj_model
    actual = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    overrides = scenarios()[job["scenario"]]
    mass, inertia, friction = nominal.body_mass.copy(), nominal.body_inertia.copy(), nominal.geom_friction.copy()
    cap = nominal.body("oakink_cap").id
    mass[cap] *= overrides.get("mass_scale", 1.)
    inertia[cap] *= overrides.get("mass_scale", 1.)
    physical = (nominal.geom_contype != 0) | (nominal.geom_conaffinity != 0)
    friction[physical] *= overrides.get("friction_scale", 1.)
    for a, b in ((actual.body_mass, mass), (actual.body_inertia, inertia), (actual.geom_friction, friction)):
        np.testing.assert_array_equal(a, b)
    for name in ("qpos0", "body_pos", "body_quat", "body_ipos", "body_iquat", "jnt_pos", "jnt_axis", "jnt_range",
                 "geom_size", "geom_pos", "geom_quat", "geom_type", "geom_contype", "mesh_vert", "mesh_face",
                 "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm", "actuator_trnid", "eq_data", "eq_type"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(nominal, name))
    assert actual.nu == 17 and actual.nmocap == 0
    assert not any(actual.joint(int(j)).name.startswith("cap_") for j in actual.actuator_trnid[:, 0])
    env._mj_model = actual
    with np.load(directory / "trajectories.npz") as packet:
        trace = dict(packet)
    assert len(trace["active"]) == len(report["seeds"])
    for i, seed in enumerate(report["seeds"]):
        _, key = jax.random.split(jax.random.PRNGKey(seed))
        expected = np.clip(np.asarray(env.initial) + .002 * np.asarray(jax.random.normal(key, (17,))), env.low, env.high)
        np.testing.assert_allclose(trace["reset_qpos"][i, 0, env.qids], expected, atol=2e-7)
    checked = audit_trace(env, trace)
    for key in ("success", "release_rate", "mean_best_hold_s", "max_penetration_mm"):
        np.testing.assert_allclose(checked[key], report["metrics"][key], atol=2e-6)
    for key in checked["episodes"]:
        np.testing.assert_allclose(checked["episodes"][key], report["independent_audit"]["episodes"][key], atol=2e-6)
    if job["baseline"] == "video":
        np.testing.assert_array_equal(trace["requested_residual"], 0)
    if job["baseline"] == "initial":
        np.testing.assert_allclose(trace["desired_ctrl"], np.broadcast_to(np.asarray(env.initial), trace["ctrl"].shape), atol=1e-6)
    return dict(passed=True, report_sha256=digest(directory / "report.json"), source_sha256=digest(Path(__file__)),
                scope="Independent job identity, source/actual model, interventions, reset seeds, success and controls")
