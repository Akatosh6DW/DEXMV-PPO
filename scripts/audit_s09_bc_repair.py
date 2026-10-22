"""Independent native replay for full-target repair policies and pose conditions."""

import argparse
import json
from pathlib import Path
import pickle

import jax
import mujoco
import numpy as np

import run_s09_job as runner
from orca_native_physics import STATE_SPEC
from run_s09_bc_conditioned import ModeConditionedGrasp
from orca_s09_bc_observation import RepairObservationGrasp
from s09_suspension import SuspendedCylinder
from retarget_s09_reference import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--reproduce-evaluation", action="store_true",
                   help="Require bitwise original-batch physics/trace reproduction instead of dense action inference")
    a = p.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    cfg = json.loads((a.run/"run_config.json").read_text())
    assert cfg["method"] == "guided" and jax.default_backend() == cfg["backend"]
    model = Path(cfg["input"])/"model.xml"
    assert digest(model) == cfg["model_sha256"]
    options = dict(approach_weight=cfg["approach_weight"], coverage_weight=cfg["coverage_weight"],
                   orientation_mode=cfg["orientation"])
    conditioned = cfg["env_config"].get("observation_variant") == "initial_pose_mode_onehot_95_101"
    repair = json.loads((a.run/"repair_config.json").read_text()) if (a.run/"repair_config.json").exists() else {}
    mode = repair.get("observation_mode", "original")
    env = (ModeConditionedGrasp(model, **options) if conditioned else
           RepairObservationGrasp(model, observation_mode=mode, **options))
    reset = jax.jit(env.reset)
    folder = a.run/f"step_{a.step:09d}"
    with np.load(folder/"preview_trajectories.npz", allow_pickle=False) as z:
        trace = dict(z)
    net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
    params = pickle.loads((folder/"params.pkl").read_bytes())
    policy = runner.networks.make_inference_fn(net)(params, deterministic=True)
    infer = jax.jit(lambda obs: policy({"state": obs}, jax.random.PRNGKey(0))[0])
    inferred = np.asarray(infer(trace["obs"]))
    dense_error = float(np.max(np.abs(inferred[trace["active"]]-trace["action"][trace["active"]])))
    if a.reproduce_evaluation:
        # Retain the dense mismatch as evidence, and reproduce the original
        # computation graph rather than relaxing its numerical threshold.
        evaluate = runner.evaluator(env, net)
        batch_size = cfg["eval_batch"]
        seed_start = cfg["eval_start"]
        offsets = (trace["seeds"]-seed_start)//batch_size*batch_size
        inferred = np.zeros_like(trace["action"])
        for offset in np.unique(offsets):
            seeds = np.arange(seed_start+offset, seed_start+offset+batch_size, dtype=np.int32)
            _, reproduced = jax.device_get(evaluate(params, seeds))
            for i in np.flatnonzero(offsets == offset):
                slot = int(trace["seeds"][i]-seeds[0])
                for key in reproduced:
                    np.testing.assert_array_equal(reproduced[key][slot], trace[key][i], err_msg=key)
                inferred[i] = reproduced["action"][slot]
    # Bind the physical trace to the actual learned checkpoint, not merely to
    # mutually consistent saved action/control arrays.
    # Replaying all frames as one dense batch changes float32 GEMM reduction
    # order versus scan(vmap(policy)). Bound normalized-action rounding at 1e-5;
    # the independent physics replay below retains its separate tight tolerances.
    np.testing.assert_allclose(inferred[trace["active"]], trace["action"][trace["active"]], atol=1e-5, rtol=0)
    policy_error = float(np.max(np.abs(inferred[trace["active"]]-trace["action"][trace["active"]])))
    recorded = {r["seed"]: r for r in json.loads((folder/"episodes.json").read_text())}
    rows = []
    for i, seed in enumerate(trace["seeds"]):
        state = jax.device_get(reset(jax.random.PRNGKey(int(seed))))
        sim = SuspendedCylinder(model.read_text())
        sim.reset(np.asarray(state.info["cube_spawn_pos"]), np.asarray(state.data.qpos[env.oid+3:env.oid+7]))
        mujoco.mj_setState(sim.model, sim.data, np.asarray(state.data.integration_words).copy().view(np.float64), STATE_SPEC)
        active = np.flatnonzero(trace["active"][i])
        low, high = sim.model.actuator_ctrlrange.T
        expected = low + (np.clip(trace["action"][i,active], -1,1)+1)*.5*(high-low)
        np.testing.assert_allclose(trace["desired_ctrl"][i,active], expected, atol=3e-7)
        np.testing.assert_array_equal(trace["requested_residual"][i], 0)
        np.testing.assert_array_equal(trace["video_base"][i], 0)
        slot = trace["obs"][i,active,95:112]
        expected_slot = np.asarray(state.obs["state"][95:112])
        np.testing.assert_array_equal(slot, np.broadcast_to(expected_slot, slot.shape))
        max_q = max_v = max_audit = 0.
        for k in active:
            row = sim.step(trace["desired_ctrl"][i,k])
            max_q = max(max_q, float(np.max(np.abs(trace["qpos"][i,k]-sim.data.qpos))))
            max_v = max(max_v, float(np.max(np.abs(trace["qvel"][i,k]-sim.data.qvel))))
            packed = np.r_[row["loads"], row["forbidden"], row["penetration_m"], row["peak_force_n"],
                row["rope_force_n"], row["alive"], row["released"], row["valid"], row["time"]]
            max_audit = max(max_audit, float(np.max(np.abs(trace["audit"][i,k]-packed))))
            assert bool(trace["alive"][i,k]) == row["alive"]
            np.testing.assert_allclose(trace["hold"][i,k], row["hold_sec"] if row["alive"] else 0, atol=5e-6)
        assert recorded[int(seed)]["success30"] == row["success30"]
        assert max_q < 3e-6 and max_v < 1e-4 and max_audit < 1e-4
        rows.append(dict(seed=int(seed), success30=row["success30"], executed_steps=len(active),
            max_qpos_error=max_q, max_qvel_error=max_v, max_audit_error=max_audit))
    result = dict(status="verified", episodes=rows, trace_sha256=digest(folder/"preview_trajectories.npz"),
        checkpoint_sha256=digest(folder/"params.pkl"), max_policy_action_error=policy_error,
        normalized_action_tolerance=1e-5,
        dense_policy_action_error=dense_error,
        exact_original_evaluation_reproduced=a.reproduce_evaluation,
        action_check="Policy supplies full absolute target; no video base or residual added",
        scope="All executed steps of saved previews, not all unsaved evaluation trajectories")
    (folder/"native_replay_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
