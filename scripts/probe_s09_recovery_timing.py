"""Matched early/late teacher interventions on previously queried training cases.

The teacher selects one bounded offset using eight seconds of simulation, then
validates it to the original 40-second endpoint. This is neither MPC nor a new
student policy. All cases, including unsuccessful searches, remain in reports.
"""

import argparse
import json
from pathlib import Path
import pickle
import shutil
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from probe_s09_recovery_teacher import (
    ROOT, SPEC, COUNTERS, capture, offset_controls, rollout, objective_value,
)
from retarget_s09_reference import digest
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_bc_phase_curriculum import KEEP
import run_s09_job as runner
from s09_suspension import SuspendedCylinder


def training_cases(rows, training_seeds):
    cases = [r for r in rows if r["search_requested"]]
    seeds = [r["seed"] for r in cases]
    if not cases or len(set(seeds)) != len(seeds):
        raise ValueError("Expected nonempty unique queried cases")
    if any(r["seed"] not in training_seeds or not r["prefix_alive"]
           or r["impulse"] != "none" for r in cases):
        raise ValueError("Only previously queried, unperturbed training cases are allowed")
    return cases


def phase_targets(state, infer, ranges):
    observations = np.tile(np.asarray(state.obs["state"]), (800, 1))
    steps = jp.arange(800, dtype=jp.float32)
    observations[:, 93] = np.asarray(jp.minimum(steps*.05/3., 1))
    observations[:, 112] = np.asarray(steps/800)
    actions = np.asarray(infer(jp.asarray(observations)))
    low, high = np.asarray(ranges, dtype=np.float32).T
    return low+(np.clip(actions, -1, 1)+np.float32(1))*np.float32(.5)*(high-low)


def fresh_prefix(sim, state, targets, prefix_steps):
    if prefix_steps not in (40, 60):
        raise ValueError("Only the predeclared 2s and 3s interventions are supported")
    sim.reset(np.asarray(state.info["cube_spawn_pos"]),
              np.asarray(state.data.qpos[sim.oid+3:sim.oid+7]))
    words = np.asarray(state.data.integration_words).copy().view(np.float64)
    mujoco.mj_setState(sim.model, sim.data, words, SPEC)
    for target in targets[:prefix_steps]:
        row = sim.step(target)
        sim.probe_control_steps = getattr(sim, "probe_control_steps", 0)+1
        if not row["alive"]:
            raise ValueError("Previously alive prefix is no longer alive")
    return capture(sim)


def verify_replay(sim, snapshot, targets):
    report, trace = rollout(sim, snapshot, targets)
    repeat, other = rollout(sim, snapshot, targets)
    if report != repeat:
        raise ValueError("Snapshot replay reports differ")
    for key in trace:
        np.testing.assert_array_equal(trace[key], other[key])
    report.update(success30=bool(trace["success30"][-1] and trace["alive"][-1]),
                  final_time_sec=float(trace["time"][-1]),
                  best_hold_sec_including_prefix=float(trace["best_hold_sec"].max()))
    return report, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise ValueError("Use CPU for this native-physics search")
    origin = json.loads((args.probe/"protocol.json").read_text())
    if not origin["failed_only"] or origin["smoke"] or origin["nominal_source"] != "phase_only_bc":
        raise ValueError("Expected original failed-state BC teacher probe")
    source = ROOT/"artifacts/orca_benchmark_suite/S09/demonstrations"
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    split = manifest.get("train_episodes", len(manifest["seeds"])*3//4)
    source_rows = json.loads((args.probe/"results.json").read_text())
    cases = training_cases(source_rows, manifest["seeds"][:split])
    checkpoint = Path(origin["bc_checkpoint"])
    model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
    cfg = json.loads((checkpoint.parent.parent/"run_config.json").read_text())
    assert digest(checkpoint) == origin["bc_checkpoint_sha256"]
    assert digest(model) == origin["model_sha256"] == manifest["model_sha256"]
    assert cfg["reference_sha256"] == manifest["reference_sha256"]
    assert digest(source/"trajectories.npz") == origin["dataset_sha256"] == manifest["dataset_sha256"]
    params = pickle.loads(checkpoint.read_bytes())
    np.testing.assert_array_equal(np.asarray(params[1]["params"]["hidden_0"]["kernel"])[~KEEP], 0)
    env = ModeConditionedGrasp(model, "pure")
    net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
    policy = runner.networks.make_inference_fn(net)(params, deterministic=True)
    infer = jax.jit(lambda obs: policy(dict(state=obs), jax.random.PRNGKey(0))[0])
    reset = jax.jit(env.reset)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output/"sources"
    sources.mkdir()
    for name in (Path(__file__).name, "probe_s09_recovery_teacher.py", "s09_suspension.py",
                 "run_s09_bc_phase_curriculum.py", "run_s09_bc_conditioned.py"):
        shutil.copyfile(ROOT/"scripts"/name, sources/name)
    if args.smoke:
        cases = cases[:1]
    horizon, iterations = (4, 0) if args.smoke else (160, 3)
    metadata = dict(kind="matched_training_teacher_timing", source_probe=str(args.probe.resolve()),
        source_protocol_sha256=digest(args.probe/"protocol.json"),
        source_results_sha256=digest(args.probe/"results.json"),
        seeds=[r["seed"] for r in cases], prefix_steps=[40, 60], search_horizon_steps=horizon,
        optimizer="scipy.optimize.differential_evolution", popsize=4, maxiter=iterations,
        search_seed=19201, zero_initial_candidate=True, historical_offset_reused=False,
        offset_bound_rad=.08, selected_actuators=list(range(7, 17)),
        duration_sec=40, policy_updates=0, smoke=args.smoke,
        checkpoint_sha256=digest(checkpoint), model_sha256=digest(model),
        dataset_sha256=manifest["dataset_sha256"], reference_sha256=manifest["reference_sha256"],
        source_sha256=digest(Path(__file__)),
        limitation="Training cases selected by historical failure, not held-out success rates; fixed offset, not MPC or DAgger")
    (args.output/"protocol.json").write_text(json.dumps(metadata, indent=2)+"\n")
    all_results, began = [], time.monotonic()
    for prefix_steps in (40, 60):
        root = args.output/f"prefix_{prefix_steps:03d}"
        probe, teacher = root/"probe", root/"teacher"
        probe.mkdir(parents=True)
        teacher.mkdir()
        # Keep the established collector interface, but preserve this probe's
        # distinct provenance and paired-case selection instead of imitating D18.
        local_origin = dict(origin, prefix_seconds=prefix_steps*.05, seeds=metadata["seeds"],
            source_sha256=metadata["source_sha256"], smoke=args.smoke,
            search_horizon_steps=horizon, verification_steps=800-prefix_steps,
            search_method=metadata["optimizer"], iterations=iterations,
            query_selection="all previously queried training cases; no new filtering",
            limitation=metadata["limitation"],
            selection="all originally queried cases, fixed before timing comparison")
        (probe/"protocol.json").write_text(json.dumps(local_origin, indent=2)+"\n")
        (probe/"results.json").write_text(json.dumps(cases, indent=2)+"\n")
        (teacher/"protocol.json").write_text(json.dumps(dict(
            source_probe=str(probe.resolve()), source_protocol_sha256=digest(probe/"protocol.json"),
            source_results_sha256=digest(probe/"results.json"), checkpoint_sha256=digest(checkpoint),
            timing_protocol_sha256=digest(args.output/"protocol.json"), policy_optimizer_updates=0,
            refine_seconds=horizon*.05, scope=metadata["limitation"]), indent=2)+"\n")
        rows, total_steps = [], 0
        for case in cases:
            seed = case["seed"]
            state = jax.device_get(reset(jax.random.PRNGKey(seed)))
            sim = SuspendedCylinder(model.read_text())
            desired = phase_targets(state, infer, sim.model.actuator_ctrlrange)
            snapshot = fresh_prefix(sim, state, desired, prefix_steps)
            if prefix_steps == 60:
                with np.load(args.probe/f"seed{seed}_none/initial_state.npz") as old:
                    np.testing.assert_array_equal(snapshot[0], old["state"])
                    np.testing.assert_array_equal(snapshot[1]["spawn"], old["spawn"])
                    np.testing.assert_array_equal([snapshot[1][k] for k in COUNTERS[1:]], old["counters"])
            folder = teacher/f"seed{seed}"
            folder.mkdir()
            selected = np.arange(7, 17)
            history, best = [], [float("inf"), np.zeros(10)]
            def evaluate(delta):
                result, _ = rollout(sim, snapshot, offset_controls(
                    desired[prefix_steps:prefix_steps+horizon].astype(float), delta,
                    selected, sim.model.actuator_ctrlrange))
                cost = objective_value(result, delta)
                history.append(dict(evaluation=len(history), cost=cost,
                                    delta=np.asarray(delta).tolist(), **result))
                if cost < best[0]:
                    best[:] = [cost, np.array(delta)]
                return cost
            evaluate(np.zeros(10))
            if not args.smoke:
                differential_evolution(evaluate, [(-.08, .08)]*10, maxiter=iterations,
                    popsize=4, seed=19201, polish=False, tol=0., x0=np.zeros(10), workers=1)
            result = dict(seed=seed, mode=case["initial_mode"], prefix_sec=prefix_steps*.05,
                          selected_delta=best[1].tolist(), search_evaluations=len(history))
            for name, change in (("unchanged_bc", np.zeros(10)), ("selected_offset", best[1])):
                targets = offset_controls(desired[prefix_steps:].astype(float), change,
                                          selected, sim.model.actuator_ctrlrange)
                report, trace = verify_replay(sim, snapshot, targets)
                result[name] = report
                np.savez_compressed(folder/f"{name}.npz", **trace)
            snapshot_dir = probe/f"seed{seed}_none"
            snapshot_dir.mkdir()
            np.savez_compressed(snapshot_dir/"initial_state.npz", state=snapshot[0],
                spawn=snapshot[1]["spawn"], counters=np.array([snapshot[1][k] for k in COUNTERS[1:]]),
                delta_velocity=np.zeros(3), selected_offset=best[1], selected_actuators=selected)
            (folder/"search.json").write_text(json.dumps(history, indent=2)+"\n")
            rows.append(result)
            total_steps += sim.probe_control_steps
            (teacher/"results.json").write_text(json.dumps(rows, indent=2)+"\n")
            print(json.dumps(result), flush=True)
        summary = dict(status="completed", prefix_sec=prefix_steps*.05, cases=len(rows),
            unchanged_successes=sum(r["unchanged_bc"]["success30"] for r in rows),
            corrected_successes=sum(r["selected_offset"]["success30"] for r in rows),
            total_native_control_steps=total_steps, total_native_physics_substeps=total_steps*sim.nsub,
            policy_updates=0, warning=metadata["limitation"])
        (teacher/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        all_results.append(summary)
    (args.output/"summary.json").write_text(json.dumps(dict(status="completed", rows=all_results,
        elapsed_sec=time.monotonic()-began, smoke=args.smoke), indent=2)+"\n")


if __name__ == "__main__":
    main()
