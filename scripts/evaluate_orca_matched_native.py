"""Evaluate an unchanged matched policy in native MuJoCo, not MJX.

This is a separately labeled transfer check. It reuses the actor observation
and residual composition functions; no stored action trajectory is replayed.
"""

import argparse
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import time

from brax.training.agents.ppo import networks
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from diagnose_orca_grasp import contact_state
from orca_guidance_recheck import score
from orca_hic53_env import check_hic53_gates
from orca_matched_env import OrcaMatchedGrasp, finger_normal_forces, touch_observation
from orca_residual_controls import compose
from prepare_hic_orca_object import digest
from train_orca_matched import initialize_matched


def native_finger_forces(env, data):
    contact = data.contact[:data.ncon]
    return finger_normal_forces(contact.geom, contact.efc_address, data.efc_force,
                                np.asarray(env.sensor_membership), cone=env.contact_cone)


def actor_control(env, net, zero_residual=False):
    if zero_residual and env.arm != "residual":
        raise ValueError("Zero-correction baseline requires a residual reference")
    inference = networks.make_inference_fn(net)
    @jax.jit
    def run(params, qpos, qvel, ctrl, actuator_force, geom_xpos, xpos, info):
        data = SimpleNamespace(qpos=qpos, qvel=qvel, ctrl=ctrl, actuator_force=actuator_force,
                               geom_xpos=geom_xpos, xpos=xpos)
        obs = env._obs(data, info)
        action = inference(params, deterministic=True)(obs, jax.random.PRNGKey(0))[0]
        if zero_residual:
            action = jp.zeros(17)
        if env.arm == "residual":
            base = env.next_base(info["step"])
            desired, residual = compose(base, action, env.residual_bound, env._lowers, env._uppers, xp=jp)
            absolute_action = env.ctrl_to_action(desired)
        else:
            base, residual = jp.zeros(17), jp.zeros(17)
            absolute_action = jp.clip(action, -1, 1)
            desired = env._lowers + (absolute_action + 1) * .5 * (env._uppers - env._lowers)
        target = env._lowers + (absolute_action + 1) * .5 * (env._uppers - env._lowers)
        applied = jp.clip(target, ctrl - .1, ctrl + .1)
        return obs["state"], action, absolute_action, base, residual, desired, applied
    return run


def push_at_step(step, force):
    """Two 200 ms world-x pulses, starting at t=12 and t=24 seconds."""
    sign = 1. if 240 <= step < 244 else (-1. if 480 <= step < 484 else 0.)
    return np.array([sign * force, 0., 0.])


def rollout(env, inference, params, seed, push_force=0., xy_jitter=.002, yaw_deg=0., noise_std=None):
    if not np.isfinite([push_force, xy_jitter, yaw_deg]).all() or xy_jitter < 0 or yaw_deg < 0:
        raise ValueError("Invalid evaluation disturbance")
    noise_std = env.tactile_noise if noise_std is None else float(noise_std)
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("Invalid evaluation sensor noise")
    model, data = env.mj_model, mujoco.MjData(env.mj_model)
    rng = np.random.default_rng(seed)
    touch_rng = np.random.default_rng(np.random.SeedSequence([seed, 43921]))
    initial = np.asarray(env._config.initial_ctrl)
    spawn = np.asarray(env._spawn_pos) + rng.uniform(-1, 1, 3) * [xy_jitter, xy_jitter, 0.]
    data.qpos[env._hand_qids] = np.clip(initial + rng.normal(0, .002, 17), env._lowers, env._uppers)
    data.qpos[env._cube_qids] = np.r_[spawn, np.asarray(env.object_quaternion)]
    if yaw_deg:
        angle_rng = np.random.default_rng(np.random.SeedSequence([seed, 55903]))
        yaw = np.deg2rad(angle_rng.uniform(-yaw_deg, yaw_deg))
        rotated = np.empty(4)
        mujoco.mju_mulQuat(rotated, np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]),
                          np.asarray(env.object_quaternion, dtype=float))
        data.qpos[env._cube_qids[3:]] = rotated
    data.ctrl[:] = initial
    support = spawn - [0, 0, env.half_height + .004]
    data.mocap_pos[0] = support
    mujoco.mj_forward(model, data)
    # A world-z rotation leaves the object's lowest z unchanged, so the original
    # support height remains correct; verify collision instead of resampling.
    initial_penetration = max([0.] + [-float(c.dist) for c in data.contact
                                      if env._cube_geom_id in (c.geom1, c.geom2)])
    reset_qpos = data.qpos.copy()
    reset_mocap = data.mocap_pos.copy()
    last_action = np.asarray(env.ctrl_to_action(jp.asarray(initial, dtype=jp.float32)))
    touch, critic_touch, filtered = np.zeros(10, np.float32), np.zeros(10, np.float32), np.zeros(5, np.float32)
    hold, records = 0, []
    for step in range(800):
        info = dict(step=np.int32(step), stable_hold_steps=np.int32(hold), cube_spawn_pos=spawn.astype(np.float32),
                    last_act=last_action, touch_obs=touch, critic_touch_obs=critic_touch)
        inputs = [getattr(data, name).astype(np.float32) for name in
                  ("qpos", "qvel", "ctrl", "actuator_force", "geom_xpos", "xpos")]
        obs, action, absolute, base, residual, desired, applied = jax.device_get(inference(params, *inputs, info))
        data.ctrl[:] = applied
        external_force = push_at_step(step, push_force)
        data.xfrc_applied[env._cube_body_id, :3] = external_force
        t = (step + 1) * .05
        data.mocap_pos[0] = support - [0, 0, np.clip((t - 2) * .2, 0, .2)]
        for _ in range(env.n_substeps):
            mujoco.mj_step(model, data)
        bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
               mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
        if any(data.warning[int(kind)].number for kind in bad):
            raise FloatingPointError("Native evaluation produced invalid physics or a silent reset")
        fingers, forbidden, _ = contact_state(model, data)
        force = native_finger_forces(env, data)
        z = float(data.xpos[env._cube_body_id, 2])
        alive = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and z >= spawn[2] - .055
        paired = fingers[0] and fingers[1:].any() and not forbidden
        stable = alive and t >= 3 and z > spawn[2] - .01 and paired
        hold = hold + 1 if stable else 0
        speed = float(np.linalg.norm(data.qvel[env._cube_dqids][:3]))
        components = dict(stable=float(stable), done=float(not alive),
                          speed=min((speed / env.speed_scale) ** 2, 4.) if t >= 3 and paired else 0.,
                          pair=float(t >= 3 and paired), hold_progress=float(stable) * min(hold * .05 / 30, 1),
                          correction=float(np.mean((residual / env.residual_bound) ** 2)))
        if "height_retention" in env.weights:
            from orca_matched_env import height_retention
            components["height_retention"] = float(height_retention(z, spawn[2], t >= 3, paired, alive, xp=np))
        records.append(dict(obs=obs, action=action, active=True, alive=alive, hold=hold * .05,
                            reward=sum(env.weights[k] * v for k, v in components.items()) * .05,
                            qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(), mocap=data.mocap_pos.copy(),
                            contacts=int(fingers.sum()), thumb=int(fingers[0]), forbidden=int(forbidden),
                            released=t >= 3, cube_z=z, spawn_z=spawn[2], cube_speed=speed,
                            video_base=base, requested_residual=residual, desired_ctrl=desired,
                            effective_residual=data.ctrl - base if env.arm == "residual" else np.zeros(17),
                            finger_normal_force=force, touch_obs=touch,
                            external_force_n=external_force,
                            initial_penetration_m=initial_penetration,
                            reset_qpos=reset_qpos, reset_mocap=reset_mocap,
                            actuator_force_rms=np.sqrt(np.mean(data.actuator_force ** 2)),
                            **{"component_" + k: v for k, v in components.items()}))
        if not alive:
            break
        touch = touch_observation(filtered, env.tactile_enabled).astype(np.float32)
        critic_touch = touch_observation(filtered, True).astype(np.float32)
        noisy = np.clip(force + noise_std * touch_rng.normal(size=5), 0, 5)
        filtered = ((1 - env.tactile_filter) * filtered + env.tactile_filter * noisy).astype(np.float32)
        last_action = absolute
    while len(records) < 800:
        records.append(dict(records[-1], active=False, alive=False, hold=0., reward=0.))
    return {k: np.asarray([r[k] for r in records]) for k in records[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed-start", type=int, default=1400)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--push-force", type=float, default=0., help="World-x force in N for paired 200 ms pulses; evaluation only")
    args = p.parse_args()
    if args.count < 1:
        p.error("Positive count required")
    if not np.isfinite(args.push_force) or not 0 <= args.push_force <= .2:
        p.error("Push force must be finite and in [0, .2] N")
    if jax.default_backend() != "cpu":
        p.error("Use JAX_PLATFORMS=cpu; native physics must not contend with GPU training")
    meta = json.loads((args.run / "run_config.json").read_text())
    model, reference, gate = [Path(meta[k]) for k in ("model", "reference", "gate")]
    check_hic53_gates(model, reference, gate)
    if digest(model) != meta["model_sha256"] or digest(reference) != meta["reference_sha256"]:
        raise ValueError("Checkpoint model/reference changed")
    if meta["actor_dim"] != 123 or meta["critic_dim"] != 140:
        raise ValueError("Not a matched-interface checkpoint")
    env = OrcaMatchedGrasp(model, "residual" if meta["method"] == "residual" else "pure",
                           reference, meta["weights"], tactile=meta["tactile"] == "on",
                           residual_bound=meta.get("residual_bound", .08),
                           reference_mode=meta.get("reference_mode", "video"))
    net, _ = initialize_matched(env, meta["seed"], meta["init_std"])
    checkpoint = args.run / f"step_{args.step:09d}/params.pkl"
    params = jax.tree_util.tree_map(np.asarray, pickle.loads(checkpoint.read_bytes()))
    inference = actor_control(env, net)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.monotonic()
    seeds = list(range(args.seed_start, args.seed_start + args.count))
    for seed in seeds:
        trace = rollout(env, inference, params, seed, push_force=args.push_force)
        assert all(np.isfinite(v).all() for v in trace.values())
        rows.append(trace)
        print("NATIVE_POLICY", seed, "hold_sec", float(trace["hold"].max()), flush=True)
    trace = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    metrics, holds, success = score(trace, env)
    final_valid = trace["alive"][:, -1] & (trace["hold"][:, -1] > 0)
    metrics.update(success15_any=float(np.mean(holds >= 15)), success15=float(np.mean((holds >= 15) & final_valid)))
    np.savez_compressed(args.output / "trajectories.npz", **trace)
    report = dict(physics_backend="native_mujoco", policy_backend="jax_cpu", method=meta["method"], tactile=meta["tactile"],
                  residual_bound=env.residual_bound,
                  reference_mode=env.reference_mode,
                  perturbation=dict(force_n=args.push_force, world_axis="x", start_times_s=[12., 24.],
                                    signs=[1, -1], duration_s=.2, source="additional evaluation challenge, not observed source-video forces"),
                  model_sha256=digest(model), reference_sha256=digest(reference), checkpoint_sha256=digest(checkpoint),
                  seeds=seeds, metrics=metrics, holds=holds.tolist(), success30=success.tolist(),
                  elapsed_sec=time.monotonic() - started,
                  reset_rng="NumPy seed; paired within native evaluations, not same initial samples as MJX",
                  observation="Shared actor observation function with float32 inputs; native double-precision physics",
                  note="Closed-loop policy transfer diagnostic; not new training or pooled GPU evaluation")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "evaluation_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
