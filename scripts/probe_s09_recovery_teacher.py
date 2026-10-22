"""Training-state recovery search prerequisite, not DAgger or a learned policy.

Replay two predeclared training demonstrations to 4s, apply a velocity impulse,
and search bounded target offsets with SciPy differential evolution. Selection
uses only a 1s horizon; a separate 4s replay tests the chosen action sequence.
No success threshold, source demonstration, or benchmark controller is changed.
"""

import argparse
import copy
import json
from pathlib import Path
import pickle
import time

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from retarget_s09_reference import digest
from s09_suspension import SuspendedCylinder

ROOT = Path(__file__).resolve().parents[1]
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
COUNTERS = ("spawn", "initial_penetration", "hold", "best", "ever_forbidden", "max_depth")


def unpack_counters(spawn, values):
    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Invalid serialized simulation counters")
    if any(x < 0 or x != int(x) for x in values[1:3]) or values[3] not in (0,1):
        raise ValueError("Hold counters must be integers and forbidden must be boolean")
    return dict(spawn=np.asarray(spawn).copy(), initial_penetration=float(values[0]),
        hold=int(values[1]), best=int(values[2]), ever_forbidden=bool(values[3]),
        max_depth=float(values[4]))


def capture(sim):
    state = np.empty(mujoco.mj_stateSize(sim.model, SPEC))
    mujoco.mj_getState(sim.model, sim.data, state, SPEC)
    return state, {key: copy.deepcopy(getattr(sim, key)) for key in COUNTERS}


def restore(sim, saved):
    # Reset warnings/derived scratch; restore warm starts and all integration data.
    mujoco.mj_resetData(sim.model, sim.data)
    mujoco.mj_setState(sim.model, sim.data, saved[0], SPEC)
    for key, value in saved[1].items():
        setattr(sim, key, copy.deepcopy(value))


def offset_controls(base, delta, selected, ranges):
    base, delta = np.asarray(base), np.asarray(delta)
    if base.ndim != 2 or base.shape[1] != 17 or delta.shape != (len(selected),):
        raise ValueError("Expected a time x 17 control sequence and selected-joint offsets")
    target = base.copy()
    target[:, selected] += delta
    return np.clip(target, *np.asarray(ranges).T)


def summarize(rows, requested_steps):
    valid = np.array([r["valid"] for r in rows], bool)
    coverage = np.array([min(r["loads"][0]/.02, 1)*min(max(r["loads"][1:])/.02, 1)
                         for r in rows])
    streak = best = 0
    for value in valid:
        streak = streak+1 if value else 0
        best = max(streak, best)
    depth = max(r["penetration_m"] for r in rows)
    forbidden = any(r["forbidden"] for r in rows)
    return dict(executed_steps=len(rows), requested_steps=requested_steps,
        alive_at_end=bool(rows[-1]["alive"]), valid_fraction=float(valid.sum()/requested_steps),
        mean_coverage=float(coverage.sum()/requested_steps),
        mean_positive_height_loss_m=float(np.maximum([r["height_loss_m"] for r in rows], 0).mean()),
        final_continuous_valid_sec=streak*.05, best_continuous_valid_sec=best*.05,
        last_second_valid=bool(streak>=20 and len(rows)==requested_steps),
        max_penetration_m=float(depth), forbidden=bool(forbidden),
        peak_force_n=float(max(r["peak_force_n"] for r in rows)),
        rope_force_n=float(max(r["rope_force_n"] for r in rows)),
        recovery_candidate=bool(streak>=20 and len(rows)==requested_steps and depth<=.003
                                and not forbidden and rows[-1]["alive"]))


def rollout(sim, saved, controls):
    restore(sim, saved)
    rows, states, actual_ctrl = [], [], []
    for target in controls:
        row = sim.step(target)
        rows.append(row)
        states.append(sim.data.qpos.copy())
        actual_ctrl.append(sim.data.ctrl.copy())
        if not row["alive"]:
            break
    trace = {key:np.asarray([row[key] for row in rows]) for key in rows[0]}
    trace.update(qpos=np.asarray(states), ctrl=np.asarray(actual_ctrl),
                 desired_ctrl=np.asarray(controls[:len(rows)]))
    sim.probe_control_steps = getattr(sim, "probe_control_steps", 0)+len(rows)
    return summarize(rows, len(controls)), trace


def objective_value(result, delta):
    return (100*(not result["alive_at_end"]) + 100*result["forbidden"]
        + 1000*max(result["max_penetration_m"]-.003, 0)
        + result["mean_positive_height_loss_m"]/.015
        + 2*(1-result["mean_coverage"]) + 2*(1-result["valid_fraction"])
        + .02*float(np.linalg.norm(delta)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bc-checkpoint", type=Path,
                        help="Optional phase-only BC for prefix/nominal actions, verified before use")
    parser.add_argument("--training-count", type=int, default=2)
    parser.add_argument("--prefix-sec", type=float, default=4.)
    parser.add_argument("--failed-only", action="store_true",
                        help="Query first six failing nominal training continuations; no external impulse")
    args = parser.parse_args()
    import jax
    from run_s09_bc_conditioned import ModeConditionedGrasp
    if jax.default_backend() != "cpu":
        raise ValueError("This native-physics diagnostic reserves GPU for training")
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT/"artifacts/orca_benchmark_suite/S09/demonstrations"
    model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    assert digest(model) == manifest["model_sha256"]
    assert digest(source/"trajectories.npz") == manifest["dataset_sha256"]
    train_count = manifest.get("train_episodes", len(manifest["seeds"])*3//4)
    if not 1 <= args.training_count <= train_count:
        parser.error("Only the existing training split may be used")
    if not 3 <= args.prefix_sec <= 10 or not np.isclose(args.prefix_sec/.05, round(args.prefix_sec/.05)):
        parser.error("Use a control-step-aligned prefix between 3 and 10 seconds")
    if args.failed_only and not args.bc_checkpoint:
        parser.error("Failed-only queries require the verified phase-only BC")
    count = 1 if args.smoke else args.training_count
    with np.load(source/"trajectories.npz") as z:
        actions = z["action"][:count].copy()
    seeds = manifest["seeds"][:count]
    prefix_steps = round(args.prefix_sec/.05)
    impulses = [("none", [0.,0.,0.]), ("x_pos", [.04,0.,0.]), ("x_neg", [-.04,0.,0.]),
                ("y_pos", [0.,.04,0.]), ("y_neg", [0.,-.04,0.]), ("z_neg", [0.,0.,-.04])]
    if args.smoke or args.failed_only:
        impulses = impulses[:1]
    env = ModeConditionedGrasp(model, "pure")
    reset = jax.jit(env.reset)
    phase_policy = None
    if args.bc_checkpoint:
        import jax.numpy as jp
        import run_s09_job as runner
        from run_s09_bc_phase_curriculum import KEEP
        cfg = json.loads((args.bc_checkpoint.parent.parent/"run_config.json").read_text())
        if cfg["model_sha256"] != digest(model) or cfg["reference_sha256"] != manifest["reference_sha256"]:
            raise ValueError("BC source scene/reference mismatch")
        if cfg["method"] != "guided" or cfg["env_config"].get("observation_variant") != "initial_pose_mode_onehot_95_101":
            raise ValueError("Only the original mode/phase BC observation interface is supported")
        params = pickle.loads(args.bc_checkpoint.read_bytes())
        kernel = np.asarray(params[1]["params"]["hidden_0"]["kernel"])
        np.testing.assert_array_equal(kernel[~KEEP], 0)
        net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
        policy = runner.networks.make_inference_fn(net)(params, deterministic=True)
        phase_policy = jax.jit(lambda obs:policy(dict(state=obs), jax.random.PRNGKey(0))[0])
    sim = SuspendedCylinder(model.read_text())
    selected = np.arange(7,17)  # Middle, index and thumb; wrist is unchanged.
    horizon, verification_steps = (4, 20) if args.smoke else (20, 80)
    iterations = 0 if args.smoke else 3
    protocol = dict(kind="training_demonstration_local_recovery_search_not_policy_training",
        dataset_sha256=digest(source/"trajectories.npz"), model_sha256=digest(model),
        seed_selection="first N training demonstrations, fixed before search", seeds=seeds,
        impulses=impulses, impulse_unit="m/s added once to object linear velocity after prefix",
        prefix_seconds=args.prefix_sec, failed_only=args.failed_only,
        query_selection="first six failed 4s nominal continuations with alive prefix" if args.failed_only else "all predeclared cases",
        selected_actuators=[sim.model.actuator(int(i)).name for i in selected],
        offset_bound_rad=.08, search_horizon_steps=horizon, verification_steps=verification_steps,
        search_method="scipy.optimize.differential_evolution", iterations=iterations, popsize=4,
        search_seed=19201, source_sha256=digest(Path(__file__)), smoke=args.smoke,
        nominal_source="phase_only_bc" if phase_policy else "recorded_reference_demonstration",
        bc_checkpoint=str(args.bc_checkpoint.resolve()) if args.bc_checkpoint else None,
        bc_checkpoint_sha256=digest(args.bc_checkpoint) if args.bc_checkpoint else None,
        limitation="Fixed offset chosen at impulse time, not receding-horizon control, DAgger, DART, BC, PPO, or a 30s benchmark. No validation/test trajectories used.")
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    all_rows, began, queried = [], time.monotonic(), 0
    for episode, seed in enumerate(seeds):
        state = jax.device_get(reset(jax.random.PRNGKey(seed)))
        if phase_policy:
            # This is valid only because all feedback input rows were verified zero.
            obs = np.tile(np.asarray(state.obs["state"]), (800, 1))
            steps = jp.arange(800, dtype=jp.float32)
            obs[:,93] = np.asarray(jp.minimum(steps*.05/3., 1))
            obs[:,112] = np.asarray(steps/800)
            actions[episode] = np.asarray(phase_policy(jp.asarray(obs)))
        sim.reset(np.asarray(state.info["cube_spawn_pos"]), np.asarray(state.data.qpos[sim.oid+3:sim.oid+7]))
        words = np.asarray(state.data.integration_words).copy().view(np.float64)
        mujoco.mj_setState(sim.model, sim.data, words, SPEC)
        low, high = sim.model.actuator_ctrlrange.astype(np.float32).T
        # This native diagnostic records its exact targets; it is not a bitwise
        # claim about the differently compiled historical collection rollout.
        desired = low + (np.clip(actions[episode], -1, 1)+np.float32(1))*np.float32(.5)*(high-low)
        prefix = []
        for target in desired[:prefix_steps]:
            prefix.append(sim.step(target))
            if not prefix[-1]["alive"]:
                break
        sim.probe_control_steps = getattr(sim, "probe_control_steps", 0)+len(prefix)
        nominal = capture(sim)
        for label, impulse in impulses:
            restore(sim, nominal)
            sim.data.qvel[sim.dof:sim.dof+3] += impulse
            perturbed = capture(sim)
            folder = args.output/f"seed{seed}_{label}"
            folder.mkdir()
            controls = desired[prefix_steps:prefix_steps+verification_steps].astype(float)
            baseline, _ = rollout(sim, perturbed, controls)
            search = not args.failed_only or (prefix[-1]["alive"] and
                not baseline["recovery_candidate"] and queried < 6)
            if search:
                queried += 1
            history = []
            best = [float("inf"), np.zeros(len(selected))]
            def evaluate(delta):
                result, _ = rollout(sim, perturbed, offset_controls(
                    controls[:horizon], delta, selected, sim.model.actuator_ctrlrange))
                cost = objective_value(result, delta)
                history.append(dict(evaluation=len(history), cost=cost,
                                    delta=np.asarray(delta).tolist(), **result))
                if cost < best[0]:
                    best[:] = [cost, np.array(delta)]
                return cost
            evaluate(np.zeros(len(selected)))
            if not args.smoke and search:
                differential_evolution(evaluate, [(-.08,.08)]*len(selected),
                    maxiter=iterations, popsize=4, seed=19201, polish=False,
                    tol=0., x0=np.zeros(len(selected)), workers=1)
            outcomes = {}
            for name, delta in (("unchanged_reference", np.zeros(len(selected))), ("selected_offset", best[1])):
                result, trace = rollout(sim, perturbed, offset_controls(
                    controls, delta, selected, sim.model.actuator_ctrlrange))
                # Restore and replay the selected sequence independently of the search.
                repeated, trace2 = rollout(sim, perturbed, trace["desired_ctrl"])
                for key in trace:
                    np.testing.assert_array_equal(trace[key], trace2[key])
                np.savez_compressed(folder/f"{name}.npz", **trace)
                outcomes[name] = result
                if not prefix[-1]["alive"]:
                    outcomes[name]["recovery_candidate"] = False
            np.savez_compressed(folder/"initial_state.npz", state=perturbed[0],
                spawn=perturbed[1]["spawn"], counters=np.array([perturbed[1][k] for k in COUNTERS[1:]]),
                delta_velocity=np.asarray(impulse), selected_offset=best[1], selected_actuators=selected)
            (folder/"search.json").write_text(json.dumps(history, indent=2)+"\n")
            row = dict(seed=seed, impulse=label, prefix_alive=all(r["alive"] for r in prefix),
                prefix_last_valid=bool(prefix[-1]["valid"]), selected_delta=best[1].tolist(),
                actual_prefix_steps=len(prefix),
                search_evaluations=len(history), search_requested=search,
                initial_mode=int(state.info["reference_index"]), replay_identical=True, **outcomes)
            all_rows.append(row)
            (args.output/"results.json").write_text(json.dumps(all_rows, indent=2)+"\n")
            print(json.dumps(row), flush=True)
    (args.output/"summary.json").write_text(json.dumps(dict(status="completed", cases=len(all_rows),
        unchanged_candidates=sum(r["unchanged_reference"]["recovery_candidate"] for r in all_rows),
        selected_candidates=sum(r["selected_offset"]["recovery_candidate"] for r in all_rows),
        searched_cases=queried, elapsed_seconds=time.monotonic()-began, policy_updates=0,
        total_native_control_steps=sim.probe_control_steps,
        total_native_physics_substeps=sim.probe_control_steps*sim.nsub,
        formal_30s_success_not_measured=True), indent=2)+"\n")


if __name__ == "__main__":
    main()
