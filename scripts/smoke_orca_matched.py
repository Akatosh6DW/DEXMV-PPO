"""GPU check that tactile on/off changes observation only, not physics/reward."""

import argparse
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jp
import numpy as np

from orca_hic53_env import check_hic53_gates
from orca_matched_env import OrcaMatchedGrasp
from prepare_hic_orca_object import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("gpu", "cpu"), default="gpu",
                   help="CPU is a runtime diagnostic only, not a GPU training gate")
    p.add_argument("--vectorized-cone-hessian", action="store_true")
    p.add_argument("--native-physics", action="store_true")
    p.add_argument("--residual-bound", type=float, default=.08)
    p.add_argument("--reference-mode", choices=("video", "endpoint", "initial"), default="video")
    p.add_argument("--compare-to", type=Path,
                   help="Require identical saved off/on/off trajectories from a previous runtime")
    args = p.parse_args()
    if args.native_physics and args.vectorized_cone_hessian:
        p.error("Native physics does not use the MJX Hessian implementation")
    if args.native_physics:
        jax.config.update("jax_pmap_shmap_merge", False)
    runtime = None
    if args.vectorized_cone_hessian:
        from orca_solver_runtime import install
        runtime = install()
    check_hic53_gates(args.model, args.reference, args.gate)
    assert jax.default_backend() == args.backend
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(Path("artifacts/orca30/jax_cache").resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "smoke_source.py").write_bytes(Path(__file__).read_bytes())
    if runtime:
        source = Path(__file__).with_name("orca_solver_runtime.py")
        (args.output / source.name).write_bytes(source.read_bytes())
        runtime["source_sha256"] = digest(source)
    outputs = []
    timings = []
    env_type = OrcaMatchedGrasp
    if args.native_physics:
        from orca_native_physics import OrcaNativeMatchedGrasp
        env_type = OrcaNativeMatchedGrasp
        source = Path(__file__).with_name("orca_native_physics.py")
        (args.output / source.name).write_bytes(source.read_bytes())
    env = env_type(args.model, "residual", args.reference, tactile=True, residual_bound=args.residual_bound,
                   reference_mode=args.reference_mode)
    @jax.jit
    def run(key, enabled):
        state = env.reset(key)
        state.info["tactile_actor_mask"] = enabled
        def step(state, _):
            nxt = env.step(state, jp.zeros(17))
            return nxt, dict(obs=nxt.obs["state"], critic=nxt.obs["privileged_state"],
                             qpos=nxt.data.qpos, qvel=nxt.data.qvel, reward=nxt.reward,
                             force=nxt.info["finger_normal_force"], done=nxt.done)
        return jax.lax.scan(step, state, None, length=100)[1]
    for enabled in (False, True, False):
        started = time.monotonic()
        result = jax.device_get(run(jax.random.PRNGKey(9100), jp.bool_(enabled)))
        timings.append(time.monotonic() - started)
        np.savez_compressed(args.output / f"rollout_{len(outputs)}.npz", **result)
        assert result["obs"].shape == (100, 123) and result["critic"].shape == (100, 140)
        assert all(np.isfinite(value).all() for value in result.values())
        np.testing.assert_array_equal(result["obs"][:, 94], 0.)
        outputs.append(result)
        print("ROLLOUT", enabled, "force max", float(result["force"].max()),
              "elapsed_sec", timings[-1], flush=True)
    off, on, repeated_off = outputs
    differences = {name: dict(on_off=float(np.max(np.abs(off[name].astype(float) - on[name]))),
                              repeat_off=float(np.max(np.abs(off[name].astype(float) - repeated_off[name]))))
                   for name in ("qpos", "qvel", "reward", "force", "done")}
    (args.output / "differences.json").write_text(json.dumps(differences, indent=2) + "\n")
    for name in ("qpos", "qvel", "reward", "force", "done"):
        np.testing.assert_allclose(off[name], on[name], atol=1e-7, rtol=0, err_msg=name)
        np.testing.assert_allclose(off[name], repeated_off[name], atol=1e-7, rtol=0, err_msg="repeat_" + name)
    np.testing.assert_allclose(off["obs"][:, :-10], on["obs"][:, :-10], atol=1e-7, rtol=0)
    np.testing.assert_allclose(off["critic"], on["critic"], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(off["obs"][:, -10:], 0.)
    # The fixed initial-pose control is allowed to fail acquisition, but masking
    # and force decoding must still be verified independently in the unit tests.
    if args.reference_mode != "initial":
        assert np.max(on["obs"][:, -10:]) > .01 and np.max(on["force"]) > .01
    if args.reference_mode == "video":
        assert not np.any(on["done"])
    bound_check = None
    if args.residual_bound != .08:
        import mujoco
        @jax.jit
        def stress(key):
            state = env.reset(key)
            def step(state, _):
                action = jp.sin(jp.arange(17) * 1.7 + state.info["step"] * .2)
                nxt = env.step(state, action)
                return nxt, dict(qpos=nxt.data.qpos, ctrl=nxt.data.ctrl,
                                 mocap=nxt.data.mocap_pos, requested=nxt.info["requested_residual"],
                                 effective=nxt.info["effective_residual"])
            return jax.lax.scan(step, state, None, length=100)[1]
        control_trace = jax.device_get(stress(jax.random.PRNGKey(9500)))
        assert all(np.isfinite(v).all() for v in control_trace.values())
        requested = float(np.abs(control_trace["requested"]).max())
        effective = float(np.abs(control_trace["effective"]).max())
        assert max(requested, effective) <= args.residual_bound + 1e-5
        assert float(np.abs(np.diff(control_trace["ctrl"], axis=0)).max()) <= .10001
        np.savez_compressed(args.output / "bound_control_trace.npz", **control_trace)
        data = mujoco.MjData(env.mj_model)
        depth = 0.
        for qpos, mocap in zip(control_trace["qpos"], control_trace["mocap"]):
            data.qpos[:] = qpos
            data.mocap_pos[:] = mocap
            mujoco.mj_forward(env.mj_model, data)
            for contact in data.contact:
                if env._cube_geom_id in (contact.geom1, contact.geom2):
                    depth = max(depth, -float(contact.dist))
        assert depth <= .005
        bound_check = dict(passed=True, steps=100, requested_max_rad=requested,
                           effective_max_rad=effective, max_penetration_m=depth,
                           scope="Finite nonzero-action probe with limits and replay distances; not an exhaustive safety proof")
    pmap_check = None
    if args.native_physics:
        from mujoco_playground._src import wrapper
        wrapped = wrapper.wrap_for_brax_training(env, episode_length=800, full_reset=True)
        devices = jax.local_devices()[:1]
        keys = jax.random.split(jax.random.PRNGKey(9400), 4)[None, ...]
        state = jax.pmap(wrapped.reset, axis_name="i", devices=devices)(keys)
        replicated = jax.device_put_replicated(jp.float32(0), devices)
        def advance(state, value):
            return wrapped.step(state, jp.zeros((4, 17))), value + 1
        state, value = jax.pmap(advance, axis_name="i", devices=devices)(state, replicated)
        jax.block_until_ready(state)
        assert np.isfinite(np.asarray(state.data.qpos)).all()
        assert float(np.asarray(value)[0]) == 1
        pmap_check = dict(pass_full_reset_wrapper=True, environments=4, devices=1)
    comparison = None
    if args.compare_to:
        baseline = json.loads((args.compare_to / "report.json").read_text())
        if (baseline["model_sha256"] != digest(args.model)
                or baseline["reference_sha256"] != digest(args.reference)
                or baseline["backend"] != jax.default_backend()
                or baseline["xla_flags"] != os.environ.get("XLA_FLAGS", "")):
            raise ValueError("Runtime comparison must preserve model, reference, backend and XLA flags")
        comparison = {}
        for i, actual in enumerate(outputs):
            with np.load(args.compare_to / f"rollout_{i}.npz") as previous:
                for name, value in actual.items():
                    comparison[f"{i}_{name}"] = float(np.max(np.abs(value - previous[name])))
        (args.output / "runtime_differences.json").write_text(json.dumps(comparison, indent=2) + "\n")
        if any(value != 0 for value in comparison.values()):
            raise AssertionError("Runtime transformation changed a saved rollout; inspect differences")
    np.savez_compressed(args.output / "trace.npz", **{f"{mode}_{k}": v for mode, data in zip(("off", "on"), outputs) for k, v in data.items()})
    report = dict(pass_paired_tactile=True, model_sha256=digest(args.model), reference_sha256=digest(args.reference),
                  gate_sha256=digest(args.gate), backend=jax.default_backend(), steps=100,
                  actor_dim=123, critic_dim=140, max_force_n=float(on["force"].max()),
                  max_qpos_difference=float(np.abs(on["qpos"] - off["qpos"]).max()),
                  contact_cone=env.contact_cone, hold_counter_masked=True,
                  xla_flags=os.environ.get("XLA_FLAGS", ""), jax_version=jax.__version__,
                  rollout_elapsed_sec=timings, first_rollout_includes_compilation=True,
                  solver_runtime=runtime,
                  physics_backend="native_mujoco_callback" if args.native_physics else "mjx_jax",
                  residual_bound=args.residual_bound, bound_control_check=bound_check,
                  reference_mode=args.reference_mode,
                  matched_source_sha256=digest(Path(__file__).with_name("orca_matched_env.py")),
                  native_source_sha256=digest(Path(__file__).with_name("orca_native_physics.py")) if args.native_physics else None,
                  pmap_shmap_merge=bool(jax.config.jax_pmap_shmap_merge), pmap_wrapper_check=pmap_check,
                  exact_runtime_comparison=comparison,
                  same_compiled_kernel=True, repeat_off_qpos_difference=differences["qpos"]["repeat_off"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
