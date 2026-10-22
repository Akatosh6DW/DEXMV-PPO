"""Bind angular-feedback checkpoints, input measurements, and native physics."""

import argparse
import json
from pathlib import Path
import pickle

import jax
import mujoco
import numpy as np

from run_s09_angular_feedback import AngularFeedbackGrasp, FEATURE, constrained
from orca_native_physics import STATE_SPEC
from retarget_s09_reference import digest
from s09_suspension import SuspendedCylinder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    cfg = json.loads((args.run/"run_config.json").read_text())
    assert cfg["env_config"]["angular_velocity_observation"] == FEATURE and cfg["method"] == "guided"
    assert jax.default_backend() == cfg["backend"]
    model = Path(cfg["input"])/"model.xml"
    assert digest(model) == cfg["model_sha256"]
    env = AngularFeedbackGrasp(model, "pure", approach_weight=cfg["approach_weight"],
        coverage_weight=cfg["coverage_weight"], orientation_mode=cfg["orientation"])
    folder = args.run/f"step_{args.step:09d}"
    with np.load(folder/"preview_trajectories.npz") as z:
        trace = dict(z)
    runner = constrained.improvement.runner
    net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
    params = pickle.loads((folder/"params.pkl").read_bytes())
    policy = runner.networks.make_inference_fn(net)(params, deterministic=True)
    infer = jax.jit(lambda obs:policy(dict(state=obs), jax.random.PRNGKey(0))[0])
    predicted = np.asarray(infer(trace["obs"]))
    np.testing.assert_allclose(predicted[trace["active"]], trace["action"][trace["active"]], atol=1e-5, rtol=0)
    action_error = float(np.max(np.abs(predicted[trace["active"]]-trace["action"][trace["active"]])))
    recorded = {r["seed"]:r for r in json.loads((folder/"episodes.json").read_text())}
    reset = jax.jit(env.reset)
    results = []
    for i, seed in enumerate(trace["seeds"]):
        state = jax.device_get(reset(jax.random.PRNGKey(int(seed))))
        sim = SuspendedCylinder(model.read_text())
        sim.reset(np.asarray(state.info["cube_spawn_pos"]), np.asarray(state.data.qpos[env.oid+3:env.oid+7]))
        mujoco.mj_setState(sim.model, sim.data, np.asarray(state.data.integration_words).copy().view(np.float64), STATE_SPEC)
        active = np.flatnonzero(trace["active"][i])
        low, high = sim.model.actuator_ctrlrange.T
        expected = low+(np.clip(trace["action"][i,active], -1,1)+1)*.5*(high-low)
        np.testing.assert_allclose(trace["desired_ctrl"][i,active], expected, atol=3e-7)
        np.testing.assert_array_equal(trace["requested_residual"][i], 0)
        np.testing.assert_array_equal(trace["video_base"][i], 0)
        mode = np.asarray(state.obs["state"][95:101])
        np.testing.assert_array_equal(trace["obs"][i,active,95:101], np.broadcast_to(mode, (len(active),6)))
        np.testing.assert_array_equal(trace["obs"][i,active,104:112], 0)
        max_q = max_v = max_audit = max_angular = 0.
        for k in active:
            observed = np.clip(sim.data.qvel[sim.dof+3:sim.dof+6]*.1, -5,5)
            max_angular = max(max_angular, float(np.max(np.abs(observed-trace["obs"][i,k,101:104]))))
            row = sim.step(trace["desired_ctrl"][i,k])
            max_q = max(max_q, float(np.max(np.abs(sim.data.qpos-trace["qpos"][i,k]))))
            max_v = max(max_v, float(np.max(np.abs(sim.data.qvel-trace["qvel"][i,k]))))
            packed = np.r_[row["loads"], row["forbidden"], row["penetration_m"], row["peak_force_n"],
                row["rope_force_n"], row["alive"], row["released"], row["valid"], row["time"]]
            max_audit = max(max_audit, float(np.max(np.abs(packed-trace["audit"][i,k]))))
            assert bool(trace["alive"][i,k]) == row["alive"]
            np.testing.assert_allclose(trace["hold"][i,k], row["hold_sec"] if row["alive"] else 0, atol=5e-6)
        assert recorded[int(seed)]["success30"] == row["success30"]
        assert max_q < 3e-6 and max_v < 1e-4 and max_audit < 1e-4 and max_angular < 1e-6
        results.append(dict(seed=int(seed), success30=row["success30"], steps=len(active),
            max_qpos_error=max_q, max_qvel_error=max_v, max_audit_error=max_audit, max_angular_input_error=max_angular))
    report = dict(status="verified", feature=FEATURE, episodes=results,
        checkpoint_sha256=digest(folder/"params.pkl"), trace_sha256=digest(folder/"preview_trajectories.npz"),
        max_policy_action_error=action_error, normalized_action_tolerance=1e-5,
        scope="All executed saved-preview steps, full absolute target, angular input bound to previous physical state; not every unsaved evaluation case")
    (folder/"native_replay_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
