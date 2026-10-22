"""Held-out cap interventions; same fixed policies, independently audited scores."""

import argparse
import json
from pathlib import Path
import pickle
import time

import jax
import mujoco
import numpy as np

from audit_oakink_cap_run import audit_trace
from oakink_cap_env import OakInkCap
from oakink_cap_protocol import ROOT, check_protocol, scenarios
from prepare_hic_orca_object import digest
from train_oakink_cap import initialize_cap, make_evaluator, summarize

SOURCES = ("evaluate_oakink_cap.py", "audit_oakink_cap_run.py", "train_oakink_cap.py",
           "oakink_cap_env.py", "oakink_cap_native.py", "oakink_cap_dynamics.py", "oakink_cap_protocol.py")


def evaluate(run, scenario, output, baseline=None, development=False):
    if jax.default_backend() != "cpu":
        raise ValueError("Independent cap evaluation runs on CPU")
    jax.config.update("jax_pmap_shmap_merge", False)
    cfg = json.loads((run / "run_config.json").read_text())
    if not development:
        protocol = check_protocol(cfg["protocol"])
        assert not cfg["development"] and json.loads((run / "formal_audit.json").read_text())["passed"]
        seeds = protocol["evaluation_seeds"]
    else:
        seeds = [19020, 19021]
    method = ("residual" if baseline == "video" else "pure") if baseline else (
        "residual" if cfg["method"] == "residual" else "pure")
    env = OakInkCap(cfg["model"], cfg["reference"], method, cfg["tactile"] == "on")
    net, params = initialize_cap(env, cfg["seed"], cfg["std"])
    final = run / f"step_{cfg['steps']:09d}/params.pkl"
    if baseline is None:
        with final.open("rb") as stream:
            params = pickle.load(stream)
    intervention = scenarios()[scenario]
    model = env.mj_model
    nominal_mass, nominal_inertia, nominal_friction = model.body_mass.copy(), model.body_inertia.copy(), model.geom_friction.copy()
    cap = model.body("oakink_cap").id
    model.body_mass[cap] *= intervention.get("mass_scale", 1.)
    model.body_inertia[cap] *= intervention.get("mass_scale", 1.)
    physical = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_friction[physical] *= intervention.get("friction_scale", 1.)
    mujoco.mj_setConst(model, mujoco.MjData(model))
    output.mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(model, str(output / "model.mjb"))
    model_hash = digest(output / "model.mjb")
    expected_mass, expected_inertia, expected_friction = nominal_mass.copy(), nominal_inertia.copy(), nominal_friction.copy()
    expected_mass[cap] *= intervention.get("mass_scale", 1.)
    expected_inertia[cap] *= intervention.get("mass_scale", 1.)
    expected_friction[physical] *= intervention.get("friction_scale", 1.)
    saved = mujoco.MjModel.from_binary_path(str(output / "model.mjb"))
    for actual, expected in ((saved.body_mass, expected_mass), (saved.body_inertia, expected_inertia),
                             (saved.geom_friction, expected_friction)):
        np.testing.assert_array_equal(actual, expected)
    start = time.monotonic()
    trace = jax.device_get(make_evaluator(env, net, seeds, teacher="initial" if baseline == "initial" else False)(params))
    checked = audit_trace(env, trace)
    if baseline == "video":
        np.testing.assert_array_equal(trace["requested_residual"], 0.)
    elif baseline == "initial":
        np.testing.assert_allclose(trace["desired_ctrl"], np.broadcast_to(np.asarray(env.initial), trace["ctrl"].shape), atol=1e-6)
    metrics = summarize(trace)
    for key in ("success", "release_rate", "mean_best_hold_s"):
        np.testing.assert_allclose(metrics[key], checked[key], atol=1e-6)
    np.savez_compressed(output / "trajectories.npz", **trace)
    report = dict(run=str(run.resolve()), train_seed=cfg["seed"], method=cfg["method"], baseline=baseline,
                  scenario=scenario, overrides=intervention, seeds=seeds, development=development,
                  protocol_sha256=cfg.get("protocol_sha256"), model_source_sha256=digest(env.xml_path),
                  model_sha256=model_hash, reference_sha256=digest(cfg["reference"]),
                  checkpoint_sha256=digest(final) if baseline is None else None,
                  trace_sha256=digest(output / "trajectories.npz"), sources={name: digest(ROOT / "scripts" / name) for name in SOURCES},
                  metrics=metrics, independent_audit=checked, seconds=time.monotonic() - start)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(run=run.name, scenario=scenario, baseline=baseline, metrics=metrics)), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--scenario", choices=tuple(scenarios()), default="nominal")
    p.add_argument("--baseline", choices=("video", "initial"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--development", action="store_true")
    args = p.parse_args()
    evaluate(args.run, args.scenario, args.output, args.baseline, args.development)


if __name__ == "__main__":
    main()
