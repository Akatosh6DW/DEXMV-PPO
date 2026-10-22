"""Paired feedback versus nominal-action replay after a common training prefix.

This is a diagnostic, not a deployable controller or held-out benchmark. Replay
recomputes the same policy on its unperturbed observations, using the same
per-step inference graph as feedback. It never uses a different teacher.
"""

import argparse
import json
from pathlib import Path
import pickle
import shutil

import jax
import jax.numpy as jp
import mujoco
import numpy as np

import run_s09_job as runner
from orca_native_physics import STATE_SPEC
from retarget_s09_reference import digest
from run_s09_bc_conditioned import ModeConditionedGrasp
from s09_suspension import SuspendedCylinder

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def impulse_state(env, state, delta):
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("Expected a finite object linear-velocity increment")
    if not delta.any():
        return state
    # Preserve double-precision integration state, sensor history and task
    # counters. Only the object's three translational velocities are changed.
    native = env.native
    with native.lock:
        data = native.restore(np.asarray(state.data.integration_words))
        data.qvel[env.dof:env.dof+3] += delta
        expected = np.empty(native.state_size, np.float64)
        mujoco.mj_getState(native.model, data, expected, STATE_SPEC)
        packed = state.data.replace(integration_words=jp.asarray(expected.view(np.uint32).copy()),
                                    qvel=jp.asarray(data.qvel.astype(np.float32)))
    # Do not re-forward geometry or recompute phase outside its compiled path:
    # neither position nor elapsed phase changed in this velocity intervention.
    actor = jp.asarray(state.obs["state"]).at[71:74].set(packed.qvel[env.dof:env.dof+3]*2)
    privileged = jp.asarray(state.obs["privileged_state"]).at[:123].set(actor)
    return state.replace(data=packed, obs=dict(state=actor, privileged_state=privileged))


def make_rollout(env, net, steps, policy_factory=None):
    factory = runner.networks.make_inference_fn(net) if policy_factory is None else policy_factory

    @jax.jit
    def evaluate(params, state, nominal_observations, replay):
        policy = factory(params, deterministic=True)

        def step(carry, index):
            state, alive = carry
            observed = jp.where(replay, nominal_observations[index], state.obs["state"])
            action = policy(dict(state=observed), jax.random.PRNGKey(0))[0]
            nxt = jax.lax.cond(alive, lambda s: env.step(s, action), lambda s: s, state)
            next_alive = alive & (nxt.done == 0)
            trace = dict(obs=state.obs["state"], action=action, active=alive, alive=next_alive,
                qpos=nxt.data.qpos, qvel=nxt.data.qvel, ctrl=nxt.data.ctrl, audit=nxt.data.audit,
                hold=jp.where(next_alive, nxt.metrics["hold_sec"], 0),
                best_hold=nxt.metrics["best_hold_sec"], success30=nxt.metrics["success30"]*next_alive,
                success15=nxt.metrics["success15"]*next_alive,
                initial_penetration=nxt.info["initial_penetration"],
                reference_index=nxt.info["reference_index"],
                requested_residual=nxt.info["requested_residual"], desired_ctrl=nxt.info["desired_ctrl"])
            return (nxt, next_alive), trace

        (last, _), trace = jax.lax.scan(step, (state, state.done == 0), jp.arange(steps))
        return last, trace
    return evaluate


def native_audit(model, initial, trace):
    sim = SuspendedCylinder(model.read_text())
    sim.reset(np.asarray(initial.info["cube_spawn_pos"]),
              np.asarray(initial.data.qpos[sim.oid+3:sim.oid+7]))
    mujoco.mj_setState(sim.model, sim.data,
        np.asarray(initial.data.integration_words).copy().view(np.float64), STATE_SPEC)
    sim.initial_penetration = float(initial.info["initial_penetration"])
    sim.hold, sim.best = int(initial.info["hold_steps"]), int(initial.info["best_steps"])
    sim.max_depth = float(initial.info["max_depth"])
    sim.ever_forbidden = bool(initial.info["ever_forbidden"])
    qerror = verror = 0.
    for i in np.flatnonzero(trace["active"]):
        row = sim.step(trace["desired_ctrl"][i])
        qerror = max(qerror, float(np.max(np.abs(sim.data.qpos-trace["qpos"][i]))))
        verror = max(verror, float(np.max(np.abs(sim.data.qvel-trace["qvel"][i]))))
    assert qerror < 3e-6 and verror < 1e-4
    assert bool(row["success30"] and row["alive"]) == bool(trace["success30"][-1])
    return dict(max_qpos_error=qerror, max_qvel_error=verror,
                native_steps=int(trace["active"].sum()))


def compare_trace(trace, nominal, oid, dof):
    active = trace["active"] & nominal["active"]
    indices = np.flatnonzero(active)
    if not len(indices):
        raise ValueError("No shared active rollout frames")
    pos = np.linalg.norm(trace["qpos"][:, oid:oid+3]-nominal["qpos"][:, oid:oid+3], axis=-1)
    vel = np.linalg.norm(trace["qvel"][:, dof:dof+3]-nominal["qvel"][:, dof:dof+3], axis=-1)
    return dict(shared_active_frames=len(indices),
        max_position_deviation_m=float(pos[active].max()),
        max_velocity_deviation_mps=float(vel[active].max()),
        position_deviation_1s_m=float(pos[19]) if active[:20].all() else None,
        position_deviation_4s_m=float(pos[79]) if active[:80].all() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert jax.default_backend() == "cpu"
    jax.config.update("jax_default_matmul_precision", "highest")
    model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
    prefix_run = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/I27_phase_s13102"
    runs = [prefix_run, BASE/"B07_feedback_early_recovery",
            BASE/"B11_feedback_continue_control", BASE/"B10_feedback_student_corrections"]
    seeds = [90523, 90539, 90554, 90561]
    impulses = [("none", [0,0,0]), ("x_pos", [.02,0,0]), ("x_neg", [-.02,0,0]),
                ("z_pos", [0,0,.02]), ("z_neg", [0,0,-.02])]
    steps = 760
    if args.smoke:
        runs, seeds, impulses, steps = runs[:1], seeds[:1], impulses[:2], 80
    manifest = json.loads((model.parent.parent/"demonstrations/dataset_manifest.json").read_text())
    split = manifest.get("train_episodes", len(manifest["seeds"])*3//4)
    assert set(seeds).issubset(manifest["seeds"][:split])
    cfg = json.loads((prefix_run/"run_config.json").read_text())
    env = ModeConditionedGrasp(model, "pure")
    net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
    prefix_params = pickle.loads((prefix_run/"step_000000000/params.pkl").read_bytes())
    prefix_rollout, evaluate = make_rollout(env, net, 40), make_rollout(env, net, steps)
    reset = jax.jit(env.reset)
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output/"sources"
    source.mkdir()
    for name in (Path(__file__).name, "orca_s09_env.py", "orca_native_physics.py", "s09_suspension.py"):
        shutil.copyfile(ROOT/"scripts"/name, source/name)
    protocol = dict(seeds=seeds, impulses=impulses, impulse_unit="m/s", prefix_steps=40,
        suffix_steps=steps, policy_updates=0, validation_or_test_used=False, smoke=args.smoke,
        model_sha256=digest(model), reference_sha256=digest(model.parent/"reference.npz"),
        source_sha256=digest(Path(__file__)), prefix_checkpoint_sha256=digest(prefix_run/"step_000000000/params.pkl"),
        replay_after_nominal_death="Hold the nominal policy's prediction on its frozen terminal state; exclude from feedback-causality comparisons",
        scope="Selected known training states, common I27 prefix; replay is a nondeployable diagnostic")
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    initial = {}
    for seed in seeds:
        start = reset(jax.random.PRNGKey(seed))
        state, trace = jax.device_get(prefix_rollout(prefix_params, start, jp.zeros((40,123)), jp.bool_(False)))
        assert trace["alive"].all() and int(state.info["step"]) == 40
        native_audit(model, start, trace)
        initial[seed] = state
        np.savez_compressed(args.output/f"prefix_seed{seed}.npz", **trace)
    rows, policy_steps, native_steps = [], 0, 0
    for run in runs:
        config = json.loads((run/"run_config.json").read_text())
        for key in ("model_sha256", "reference_sha256", "env_config", "method"):
            assert config[key] == cfg[key], key
        checkpoint = run/"step_000000000/params.pkl"
        params = pickle.loads(checkpoint.read_bytes())
        folder = args.output/run.name
        folder.mkdir()
        for seed in seeds:
            state = initial[seed]
            _, nominal = jax.device_get(evaluate(params, state, jp.zeros((steps,123)), jp.bool_(False)))
            for label, delta in impulses:
                changed = impulse_state(env, state, delta)
                for replay in (False, True):
                    if label == "none" and not replay:
                        trace = nominal
                    else:
                        _, trace = jax.device_get(evaluate(params, changed, jp.asarray(nominal["obs"]), jp.bool_(replay)))
                    if label == "none" or run == prefix_run:
                        if label == "none":
                            for key in trace:
                                np.testing.assert_array_equal(trace[key], nominal[key])
                        elif replay:
                            # Only compare while the nominal recording is live;
                            # after nominal death the replayed phase is frozen.
                            shared = nominal["active"] & trace["active"] & closed_trace["active"]
                            for key in trace:
                                np.testing.assert_array_equal(trace[key][shared], closed_trace[key][shared])
                    if not replay:
                        closed_trace = trace
                    audit = native_audit(model, changed, trace)
                    _, episodes = runner.summarize({k:v[None] for k,v in trace.items()}, [seed])
                    path = folder/f"seed{seed}_{label}_{'replay' if replay else 'feedback'}.npz"
                    np.savez_compressed(path, **trace)
                    record = dict(run=run.name, impulse=label, replay=replay,
                        nominal_alive_at_end=bool(nominal["alive"][-1]),
                        nominal_duration_sec=float(nominal["active"].sum()*.05),
                        checkpoint_sha256=digest(checkpoint), trace_sha256=digest(path),
                        **episodes[0], **audit, **compare_trace(trace, nominal, env.oid, env.dof))
                    rows.append(record)
                    policy_steps += int(trace["active"].sum())
                    native_steps += audit["native_steps"]
                    print(json.dumps(record), flush=True)
            (args.output/"results.json").write_text(json.dumps(rows, indent=2)+"\n")
    (args.output/"summary.json").write_text(json.dumps(dict(status="completed", smoke=args.smoke,
        cases=len(rows), suffix_control_steps=policy_steps, suffix_native_validation_steps=native_steps,
        prefix_control_steps=40*len(seeds), prefix_native_validation_steps=40*len(seeds),
        policy_updates=0, scope=protocol["scope"]), indent=2)+"\n")


if __name__ == "__main__":
    main()
