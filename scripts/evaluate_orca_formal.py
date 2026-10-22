"""Paired held-out sensitivity tests for frozen policies, without retraining."""

import argparse
import json
from pathlib import Path
import pickle
import time

import jax
import mujoco
import numpy as np

from audit_orca_trace_geometry import audit as geometry_audit
from evaluate_orca_matched_native import actor_control, rollout
from orca_formal_protocol import scenarios
from orca_guidance_recheck import score
from orca_hic53_env import check_hic53_gates
from orca_matched_env import OrcaMatchedGrasp
from prepare_hic_orca_object import digest
from train_orca_matched import initialize_matched

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SOURCES = ("evaluate_orca_formal.py", "evaluate_orca_matched_native.py", "orca_formal_protocol.py",
                      "orca_matched_env.py", "orca_residual_metrics.py", "orca_guidance_recheck.py",
                      "diagnose_orca_grasp.py", "audit_orca_trace_geometry.py", "audit_orca_formal_evaluation.py")


def change_physics(model, scenario):
    """Modify native model constants, not an unused MJX copy."""
    mass = scenario.get("mass_scale", 1.)
    friction = scenario.get("friction_scale", 1.)
    if not np.isfinite([mass, friction]).all() or min(mass, friction) <= 0:
        raise ValueError("Positive finite physics multipliers required")
    cube = model.body("cube").id
    previous_mass = model.body_mass.copy()
    previous_inertia = model.body_inertia.copy()
    previous_friction = model.geom_friction.copy()
    model.body_mass[cube] *= mass
    model.body_inertia[cube] *= mass
    physical = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_friction[physical] *= friction
    if mass != 1.:
        mujoco.mj_setConst(model, mujoco.MjData(model))
    np.testing.assert_array_equal(np.delete(model.body_mass, cube), np.delete(previous_mass, cube))
    np.testing.assert_array_equal(np.delete(model.body_inertia, cube, axis=0), np.delete(previous_inertia, cube, axis=0))
    return dict(mass_scale=mass, contact_friction_scale=friction, mass_kg=float(model.body_mass[cube]),
                inertia_kg_m2=model.body_inertia[cube].tolist(),
                previous_mass_kg=float(previous_mass[cube]), previous_inertia_kg_m2=previous_inertia[cube].tolist(),
                physical_geom_ids=np.flatnonzero(physical).tolist(),
                nominal_geom_friction=previous_friction.tolist(), actual_geom_friction=model.geom_friction.tolist())


def episode_statistics(trace, env):
    metrics, holds, success = score(trace, env)
    initial_depth = trace["initial_penetration_m"][:, 0]
    valid_initial = initial_depth <= .003
    strict = success & valid_initial
    final = trace["alive"][:, -1] & (trace["hold"][:, -1] > 0)
    force = trace["finger_normal_force"]
    max_force = np.max(np.where(trace["active"][..., None], force, 0), axis=(1, 2))
    acquisition = np.any((trace["hold"] > 0) & trace["active"], axis=1)
    steps_active = trace["active"].sum(axis=1)
    metrics.update(raw_success30=metrics["success30"], success30=float(strict.mean()),
                   strict_mean_hold=float(np.where(valid_initial, holds, 0.).mean()),
                   valid_initial_count=int(valid_initial.sum()), invalid_initial_count=int((~valid_initial).sum()),
                   success30_valid_initial=float(strict.sum() / valid_initial.sum()) if valid_initial.any() else None,
                   success15=float(np.mean((holds >= 15) & final & valid_initial)),
                   success15_any=float(np.mean(holds >= 15)), valid_final=float(final.mean()),
                   acquisition_rate=float(acquisition.mean()), mean_peak_force_n=float(max_force.mean()),
                   peak_force_n=float(max_force.max()))
    episodes = dict(success30=strict.tolist(), raw_success30=success.tolist(), hold_sec=holds.tolist(),
                    final_valid=final.tolist(), valid_initial=valid_initial.tolist(),
                    initial_penetration_m=initial_depth.tolist(), max_force_n=max_force.tolist(),
                    termination_time_s=(steps_active * .05).tolist(), acquisition=acquisition.tolist())
    return metrics, episodes


def evaluate(run, step, name, output, seeds, zero_mode=None, development=False):
    started = time.monotonic()
    config = json.loads((run / "run_config.json").read_text())
    if not development:
        if not (run / "formal_audit.json").is_file():
            raise ValueError("Formal held-out tests require an audited formal run")
        assert json.loads((run / "formal_audit.json").read_text())["passed"]
        assert seeds == list(range(25000, 25064))
        assert step == (0 if config["method"] == "bc" else 655360)
    model_path, reference, gate = [ROOT / config[k] for k in ("model", "reference", "gate")]
    check_hic53_gates(model_path, reference, gate)
    assert digest(model_path) == config["model_sha256"] and digest(reference) == config["reference_sha256"]
    if zero_mode and config["method"] != "residual":
        raise ValueError("Zero-correction controls need a residual reference")
    scenario = scenarios()[name]
    env = OrcaMatchedGrasp(model_path, "residual" if config["method"] == "residual" else "pure",
                           reference, config["weights"], tactile=config["tactile"] == "on",
                           residual_bound=config.get("residual_bound", .08),
                           reference_mode=zero_mode or config.get("reference_mode", "video"))
    changes = change_physics(env.mj_model, scenario)
    net, _ = initialize_matched(env, config["seed"], config["init_std"])
    checkpoint = run / f"step_{step:09d}/params.pkl"
    params = jax.tree_util.tree_map(np.asarray, pickle.loads(checkpoint.read_bytes()))
    inference = actor_control(env, net, zero_residual=bool(zero_mode))
    output.mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(env.mj_model, str(output / "model.mjb"))
    mujoco.mj_saveLastXML(str(output / "model.xml"), env.mj_model)
    rows = []
    for seed in seeds:
        row = rollout(env, inference, params, seed,
                      push_force=scenario.get("push_force_n", 0.), xy_jitter=scenario.get("xy_jitter_m", .002),
                      yaw_deg=scenario.get("yaw_deg", 0.), noise_std=scenario.get("noise_std_n", .02))
        assert all(np.isfinite(v).all() for v in row.values())
        rows.append(row)
    trace = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    metrics, episodes = episode_statistics(trace, env)
    np.savez_compressed(output / "trajectories.npz", **trace)
    geometry = geometry_audit(output / "model.xml", output / "trajectories.npz")
    (output / "geometry_audit.json").write_text(json.dumps(geometry, indent=2) + "\n")
    report = dict(run=str(run.resolve()), train_seed=config["seed"], method=config["method"],
                  tactile=config["tactile"], reference_mode=env.reference_mode,
                  residual_bound=env.residual_bound, scenario=name, overrides=scenario,
                  zero_correction_baseline=bool(zero_mode), development_smoke=development,
                  checkpoint_sha256=digest(checkpoint), reference_sha256=digest(reference),
                  nominal_model_sha256=digest(model_path), actual_model_sha256=digest(output / "model.mjb"),
                  model_xml_sha256=digest(output / "model.xml"), physics_changes=changes,
                  trace_sha256=digest(output / "trajectories.npz"), seeds=seeds, metrics=metrics, episodes=episodes,
                  geometry=geometry, elapsed_sec=time.monotonic() - started,
                  source_sha256={n: digest(ROOT / "scripts" / n) for n in EVALUATION_SOURCES},
                  amendment_sha256=digest(ROOT / "docs/orca_formal_evaluation_amendment.md"),
                  note="Native closed-loop policy evaluation; no retraining; independent NumPy reset seeds")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(run=run.name, scenario=name, metrics=metrics, seconds=report["elapsed_sec"])), flush=True)
    jax.clear_caches()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=tuple(scenarios()), default="nominal")
    parser.add_argument("--zero-reference", choices=("video", "endpoint", "initial"))
    parser.add_argument("--development-smoke", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        parser.error("Use JAX_PLATFORMS=cpu for the independent native evaluator")
    seeds = [19000, 19001] if args.development_smoke else list(range(25000, 25064))
    evaluate(args.run, args.step, args.scenario, args.output, seeds, args.zero_reference, args.development_smoke)


if __name__ == "__main__":
    main()
