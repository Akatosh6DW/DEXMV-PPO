#!/usr/bin/env python3
"""Independent, no-autoreset evaluation of continuous unsupported cube grasp."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training import networks
from brax.training.acme import running_statistics
from mujoco_playground import registry


def merge(config, values):
    for key, value in values.items():
        if isinstance(value, dict):
            merge(config[key], value)
        else:
            config[key] = value


def max_run(mask, dt):
    best = current = 0
    for valid in mask:
        current = current + 1 if valid else 0
        best = max(best, current)
    return best * dt


def load_policy(path):
    # This installed Brax loader indexes optional None initializers as names.
    # Decode the recorded configuration without modifying checkpoint files.
    config = json.loads((path / "ppo_network_config.json").read_text())
    kwargs = config["network_factory_kwargs"]
    kwargs["activation"] = networks.ACTIVATION[kwargs["activation"]]
    for key in list(kwargs):
        if key.endswith("kernel_init_fn") and kwargs[key] is not None:
            kwargs[key] = networks.KERNEL_INITIALIZER[kwargs[key]]
    sizes = {k: v["shape"][-1] if isinstance(v, dict) else v for k, v in config["observation_size"].items()}
    normalize = running_statistics.normalize if config["normalize_observations"] else lambda obs, _: obs
    network = ppo_networks.make_ppo_networks(sizes, config["action_size"], preprocess_observations_fn=normalize, **kwargs)
    return ppo_networks.make_inference_fn(network)(checkpoint.load(str(path.resolve())), deterministic=True)


def render(env, trajectory, output, seed):
    import mediapy
    import mujoco
    from PIL import Image, ImageDraw
    data = mujoco.MjData(env.mj_model)
    renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
    frames = []
    for i in range(len(trajectory["time"])):
        data.qpos[:] = trajectory["qpos"][i]
        data.qvel[:] = trajectory["qvel"][i]
        data.mocap_pos[:] = trajectory["mocap_pos"][i]
        data.mocap_quat[:] = trajectory["mocap_quat"][i]
        data.ctrl[:] = trajectory["ctrl"][i]
        data.time = trajectory["time"][i]
        mujoco.mj_forward(env.mj_model, data)
        renderer.update_scene(data, camera="side")
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 640, 47), fill=(20, 20, 20))
        draw.text((10, 8), "HIC20 + DexMV BC + Residual PPO | seed %d | t=%.2fs" % (seed, data.time), fill="white")
        draw.text((10, 28), "support_contact=%d palm_contact=%d primary_tips=%d valid_hold=%d" % (
            trajectory["support_contact"][i], trajectory["palm_contact"][i],
            trajectory["primary_count"][i], trajectory["strict_hold"][i]), fill="white")
        frames.append(np.asarray(img))
    renderer.close()
    mediapy.write_video(str(output), frames, fps=round(1 / env.dt))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 108)))
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--render", action="store_true")
    p.add_argument("--blend", type=float, default=None, help="Runtime prior ablation; weights unchanged")
    a = p.parse_args()
    env_name = "AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect"
    config = registry.get_default_config(env_name)
    # Use the actual checkpoint task settings, then the explicit experiment overlay.
    saved = a.checkpoint / "config.json"
    if not saved.exists():
        saved = a.checkpoint.parent / "config.json"
    if saved.exists():
        merge(config, json.loads(saved.read_text()))
    merge(config, json.loads(a.env_config.read_text()))
    if a.blend is not None:
        if not 0 <= a.blend <= 1:
            raise ValueError("blend must be in [0, 1]")
        config.residual_prior.blend_alpha = a.blend
    config.impl = "jax"
    env = registry.load(env_name, config=config)
    policy = load_policy(a.checkpoint)
    support_gids = [i for i in range(env.mj_model.ngeom) if "support" in (env.mj_model.geom(i).name or "")]
    if not support_gids:
        raise ValueError("Missing support collision geometry; cannot certify unsupported grasp")
    print("devices", jax.devices(), "support geoms", support_gids, flush=True)

    def rollout(key):
        state = env.reset(key)
        def step(carry, _):
            state, key, alive = carry
            key, action_key = jax.random.split(key)
            action = policy(state.obs, action_key)[0]
            state = env.step(state, action)
            tips = env._get_tip_contact_flags(state.data)
            support = env._has_cube_contact(state.data, support_gids)
            palm = env._has_cube_contact(state.data, env._palm_contact_gids)
            primary_count = tips[0] + tips[1] + tips[4]
            cube = env.get_cube_position(state.data)
            finite = jp.all(jp.isfinite(state.data.qpos)) & jp.all(jp.isfinite(state.data.qvel))
            alive = alive & ~state.done.astype(bool) & finite
            strict = (alive & state.info["support_released"] & (support == 0) & (palm == 0)
                      & (tips[4] > 0) & (primary_count >= 2) & (cube[2] > env._spawn_z - .010))
            record = {"time": state.data.time, "strict_hold": strict, "alive": alive,
                      "support_contact": support, "palm_contact": palm, "primary_count": primary_count,
                      "cube_z": cube[2], "cube_speed": jp.linalg.norm(env.get_cube_linvel(state.data)),
                      "env_hold_sec": state.info["stable_hold_steps"] * env.dt,
                      "support_released": state.info["support_released"],
                      "qpos": state.data.qpos, "qvel": state.data.qvel, "ctrl": state.data.ctrl,
                      "mocap_pos": state.data.mocap_pos, "mocap_quat": state.data.mocap_quat}
            return (state, key, alive), record
        return jax.lax.scan(step, (state, key, jp.array(True)), None, length=a.steps)[1]

    print("Compiling independent rollout", flush=True)
    traces = jax.jit(jax.vmap(rollout))(jp.stack([jax.random.PRNGKey(s) for s in a.seeds]))
    traces = jax.device_get(traces)
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output / "trajectories.npz", **traces, seeds=np.array(a.seeds))
    rows = []
    for i, seed in enumerate(a.seeds):
        tr = {k: v[i] for k, v in traces.items()}
        duration = max_run(tr["strict_hold"], env.dt)
        rows.append({"seed": seed, "continuous_unsupported_sec": duration, "success_gt20s": duration > 20,
                     "env_max_hold_sec": float(np.max(np.where(tr["alive"], tr["env_hold_sec"], 0))),
                     "terminated": bool(np.any(~tr["alive"])),
                     "support_contact_after_release_steps": int(np.sum(tr["support_contact"] * tr["support_released"] * tr["alive"])),
                     "palm_contact_alive_steps": int(np.sum(tr["palm_contact"] * tr["alive"]))})
    with (a.output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prior_path = Path(config.residual_prior.dataset_path)
    report = {"checkpoint": str(a.checkpoint.resolve()), "prior_path": str(prior_path),
              "prior_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
              "criterion": "continuous >20s; no autoreset; alive; released; no support/palm contact; thumb and >=1 index/middle tip; height > spawn-1cm",
              "blend_alpha": config.residual_prior.blend_alpha,
              "dt": env.dt, "episodes": rows, "success_rate": np.mean([r["success_gt20s"] for r in rows]).item(),
              "mean_continuous_unsupported_sec": float(np.mean([r["continuous_unsupported_sec"] for r in rows]))}
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "env_config.json").write_text(config.to_json(indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if a.render:
        best = int(np.argmax([r["continuous_unsupported_sec"] for r in rows]))
        render(env, {k: v[best] for k, v in traces.items()}, a.output / "best_rollout.mp4", a.seeds[best])


if __name__ == "__main__":
    main()
