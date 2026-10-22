#!/usr/bin/env python3
"""Check the geometric controller in the actual MJX benchmark, with traces."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jp
import numpy as np

from orca_benchmark_env import OrcaBenchmark, default_config


def hic_closure_phase(source):
    with np.load(source) as z:
        joints = z["hand_joints"][:61]
    pinch = np.linalg.norm(joints[:, 4] - joints[:, 8], axis=1)
    span = pinch[0] - np.min(pinch)
    if span <= 1e-4:
        raise ValueError("HIC segment does not contain measurable pinch closure")
    return np.maximum.accumulate(np.clip((pinch[0] - pinch) / span, 0, 1))


def dexmv_closure_phase(source):
    with np.load(source) as z:
        tips = z["fitted_positions"][:96, [3, 6]]
    pinch = np.linalg.norm(tips[:, 0] - tips[:, 1], axis=1)
    span = pinch[0] - np.min(pinch)
    if span <= 1e-4:
        raise ValueError("DexMV retarget has no measurable pinch closure")
    return np.maximum.accumulate(np.clip((pinch[0] - pinch) / span, 0, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=Path, default=Path("artifacts/orca30/geometry_o04/report.json"))
    p.add_argument("--output", type=Path, default=Path("artifacts/orca30/mjx_o05"))
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--timing", choices=("linear", "hic_pinch", "dexmv_pinch"), default="linear")
    p.add_argument("--hic", type=Path, default=Path("artifacts/cube20/hic20/prepared/human_cube.npz"))
    p.add_argument("--dexmv", type=Path, default=Path("artifacts/cube20/hic20/dexmv_retarget/adroit_cube.npz"))
    p.add_argument("--reference", type=Path)
    p.add_argument("--episode-steps", type=int, default=800)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    best = json.loads(args.geometry.read_text())["best"]
    cfg = default_config()
    cfg.spawn_config.cube_pos = best["cube"]
    cfg.initial_ctrl[0] = best["ctrl"][0]
    env = OrcaBenchmark(cfg)
    target = jp.array(best["ctrl"])
    initial = jp.array(cfg.initial_ctrl)
    reference = None
    if args.reference:
        with np.load(args.reference) as z:
            reference = jp.array(z["ctrl"], dtype=jp.float32)
            np.testing.assert_allclose(np.diff(z["time"]), .05, atol=1e-6)
    if args.timing == "dexmv_pinch":
        phase = jp.array(dexmv_closure_phase(args.dexmv))
    else:
        phase = jp.array(hic_closure_phase(args.hic) if args.timing == "hic_pinch" else [0., 1.])

    def run(seed):
        state = env.reset(jax.random.PRNGKey(seed))
        def step(carry, _):
            state, alive = carry
            alpha = jp.clip((state.info["step"] + 1) * env.dt / 1.5, 0, 1)
            alpha = jp.interp(alpha, jp.linspace(0, 1, len(phase)), phase)
            action = env.ctrl_to_action(initial + alpha * (target - initial))
            if reference is not None:
                index = jp.minimum(state.info["step"] + 1, len(reference) - 1)
                action = env.ctrl_to_action(reference[index])
            nxt = env.step(state, action)
            alive &= nxt.done == 0
            trace = dict(obs=state.obs["state"], action=action, qpos=nxt.data.qpos,
                         ctrl=nxt.data.ctrl, mocap=nxt.data.mocap_pos,
                         alive=alive, **nxt.metrics)
            return (nxt, alive), trace
        return jax.lax.scan(step, (state, jp.bool_(True)), None, length=args.episode_steps)[1]

    started = time.monotonic()
    result = jax.jit(jax.vmap(run))(jp.arange(args.seeds))
    result = jax.device_get(result)
    np.savez_compressed(args.output / "trajectories.npz", **result)
    holds = np.max(np.where(result["alive"], result["hold_sec"], 0), axis=1)
    report = dict(kind="MJX_scripted_not_PPO", hold_sec=holds.tolist(),
                  success30=float(np.mean(holds >= 30)), elapsed_sec=time.monotonic() - started,
                  obs_size=int(result["obs"].shape[-1]), timing=args.timing,
                  guidance=f"{args.timing} + geometry-optimized endpoint",
                  geometry_sha256=hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
                  hic_sha256=hashlib.sha256(args.hic.read_bytes()).hexdigest() if args.timing != "linear" else None,
                  dexmv_sha256=hashlib.sha256(args.dexmv.read_bytes()).hexdigest() if args.timing == "dexmv_pinch" else None,
                  source_frames=[0, len(phase) - 1] if args.timing != "linear" else None,
                  retimed_closure_sec=1.5)
    if args.reference:
        report.update(guidance="full 3D DexMV trajectory, geometry-calibrated endpoint, acquisition ramp",
                      reference=str(args.reference), reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                      reference_provenance=json.loads((args.reference.parent / "report.json").read_text()))
    report["episode_steps"] = args.episode_steps
    report["alive_at_end"] = result["alive"][:, -1].tolist()
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "env_config.json").write_text(cfg.to_json(indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
