"""Development gate for cap callbacks, tactile masking and Brax resets."""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground._src import wrapper
from oakink_cap_env import OakInkCap
from prepare_hic_orca_object import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    args = p.parse_args()
    assert jax.default_backend() == args.backend
    jax.config.update("jax_pmap_shmap_merge", False)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in ("smoke_oakink_cap.py", "oakink_cap_env.py", "oakink_cap_native.py", "oakink_cap_dynamics.py"):
        path = Path(__file__).with_name(name)
        (args.output / name).write_bytes(path.read_bytes())
        sources[name] = digest(path)
    env = OakInkCap(args.model, args.reference)

    @jax.jit
    def run(key, enabled, amplitude):
        state = env.reset(key)
        state.info["tactile_mask"] = enabled

        def step(state, _):
            action = amplitude * jp.sin(jp.arange(17) * 1.7 + state.info["step"] * .2)
            nxt = env.step(state, action)
            return nxt, dict(obs=nxt.obs["state"], critic=nxt.obs["privileged_state"],
                             qpos=nxt.data.qpos, qvel=nxt.data.qvel, ctrl=nxt.data.ctrl,
                             force=nxt.data.finger_force, reward=nxt.reward, done=nxt.done,
                             base=nxt.info["video_base"], requested=nxt.info["requested_residual"],
                             effective=nxt.info["effective_residual"], penetration=nxt.data.hand_penetration,
                             released=nxt.metrics["released"], success=nxt.metrics["success"])
        return jax.lax.scan(step, state, None, length=400)[1]

    outputs, elapsed = [], []
    for enabled in (False, True, False):
        start = time.monotonic()
        trace = jax.device_get(run(jax.random.PRNGKey(18041), jp.bool_(enabled), jp.float32(0)))
        elapsed.append(time.monotonic() - start)
        assert all(np.isfinite(v).all() for v in trace.values())
        np.savez_compressed(args.output / f"paired_{len(outputs)}.npz", **trace)
        outputs.append(trace)
        print("paired", enabled, "force", float(trace["force"].max()), "seconds", elapsed[-1], flush=True)
    off, on, repeated = outputs
    assert on["obs"].shape == (400, 124) and on["critic"].shape == (400, 148)
    for key in off:
        np.testing.assert_array_equal(off[key], repeated[key], err_msg="repeat " + key)
        if key != "obs":
            np.testing.assert_array_equal(off[key], on[key], err_msg="mask " + key)
    np.testing.assert_array_equal(off["obs"][:, :-10], on["obs"][:, :-10])
    np.testing.assert_array_equal(off["obs"][:, -10:], 0)
    assert on["force"].max() > .01 and on["obs"][:, -10:].max() > .01
    stress = jax.device_get(run(jax.random.PRNGKey(18042), jp.bool_(True), jp.float32(1)))
    assert all(np.isfinite(v).all() for v in stress.values())
    np.savez_compressed(args.output / "stress.npz", **stress)
    for field in ("requested", "effective"):
        assert np.abs(stress[field]).max() <= env.bound + 1e-5, field
    slew = float(np.abs(np.diff(np.vstack([np.asarray(env.initial), stress["ctrl"]]), axis=0)).max())
    assert slew <= .10001
    assert (stress["ctrl"] >= np.asarray(env.low) - 1e-6).all()
    assert (stress["ctrl"] <= np.asarray(env.high) + 1e-6).all()

    # A short episode forces real autoresets, not only a single vectorized step.
    wrapped = wrapper.wrap_for_brax_training(env, episode_length=3, full_reset=True)
    devices = jax.local_devices()[:1]
    keys = jax.random.split(jax.random.PRNGKey(18043), 4)[None, ...]
    reset = jax.pmap(wrapped.reset, axis_name="i", devices=devices)
    state = reset(keys)
    replicated = jax.device_put_replicated(jp.float32(0), devices)

    def advance(state, value):
        return wrapped.step(state, jp.zeros((4, 17))), value + 1
    advance = jax.pmap(advance, axis_name="i", devices=devices)
    dones = []
    for _ in range(7):
        state, replicated = advance(state, replicated)
        jax.block_until_ready(state)
        assert np.isfinite(np.asarray(state.data.qpos)).all()
        dones.append(np.asarray(state.done).copy())
    assert float(np.asarray(replicated)[0]) == 7
    assert np.asarray(dones)[2].all() and np.asarray(dones)[5].all()
    assert not np.asarray(dones)[3].any()
    report = dict(passed=True, backend=args.backend, model_sha256=digest(args.model / "model.xml"),
                  reference_sha256=digest(args.reference), sources=sources, paired_exact=True,
                  actor_dim=124, critic_dim=148, rollout_seconds=elapsed, pmap_envs=4,
                  pmap_steps=7, autoreset_verified=True, max_force_n=float(on["force"].max()),
                  zero_correction_success=bool(on["success"].any()),
                  stress_requested_rad=float(np.abs(stress["requested"]).max()),
                  stress_effective_rad=float(np.abs(stress["effective"]).max()), max_ctrl_delta=slew,
                  stress_max_penetration_mm=float(stress["penetration"].max() * 1000),
                  scope="Interface and finite-control gate, not task success or exhaustive collision safety")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
