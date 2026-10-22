"""Bind base, delta, full-command history and physics of the new residual arm."""

import argparse
import json
from pathlib import Path
import pickle

import run_s09_learned_base_residual as learned
from audit_s09_wide_actor import matched_actor
from orca_native_physics import STATE_SPEC
from retarget_s09_reference import digest
from s09_suspension import SuspendedCylinder
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.networks import FeedForwardNetwork


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--step",type=int,default=0)
    args = p.parse_args()
    jax.config.update("jax_default_matmul_precision","highest")
    cfg = json.loads((args.run/"run_config.json").read_text())
    protocol = json.loads((args.run/"learned_base_config.json").read_text())
    assert cfg["policy_architecture"] == learned.ARCHITECTURE and cfg["method"] == "residual"
    assert cfg["backend"] == jax.default_backend()
    model = Path(cfg["input"])/"model.xml"
    base_path = Path(protocol["base_checkpoint"])
    assert digest(base_path) == protocol["base_sha256"] and digest(model) == cfg["model_sha256"]
    for name,value in cfg["sources"].items():
        assert digest(args.run/"sources"/name) == value
    assert digest(Path(learned.__file__)) == cfg["sources"][Path(learned.__file__).name]
    base,base_params,_ = learned.load_base(base_path,model)
    env = learned.LearnedBaseGrasp(model,base_action=base,residual_bound=protocol["residual_bound_rad"],
        approach_weight=cfg["approach_weight"],coverage_weight=cfg["coverage_weight"],orientation_mode=cfg["orientation"])
    net,_ = learned.initialize_residual(learned.residual_factory(base),base_params,cfg["seed"],cfg["std"])
    folder = args.run/f"step_{args.step:09d}"
    with np.load(folder/"preview_trajectories.npz",allow_pickle=False) as z:
        trace = dict(z)
    assert "video_base" not in trace and "learned_base" in trace
    params = pickle.loads((folder/"params.pkl").read_bytes())
    actor = matched_actor(net.policy_network,cfg["eval_batch"])
    base_actor = matched_actor(FeedForwardNetwork(init=lambda key:None,
        apply=lambda norm,params,obs:base(obs["state"])),cfg["eval_batch"])
    infer = jax.jit(lambda obs:jp.tanh(actor.apply(params[0],params[1],dict(state=obs))[...,:17]))
    infer_base = jax.jit(lambda obs:base_actor.apply(None,None,dict(state=obs)))
    predicted = np.asarray(infer(trace["obs"]))
    base_actions = np.asarray(infer_base(trace["obs"]))
    active = trace["active"].astype(bool)
    np.testing.assert_allclose(predicted[active],trace["action"][active],atol=1e-5,rtol=0)
    low,high = env.mj_model.actuator_ctrlrange.T
    expected_base = low+(base_actions+1)*.5*(high-low)
    np.testing.assert_allclose(expected_base[active],trace["learned_base"][active],atol=2e-5,rtol=0)
    full,requested = jax.jit(lambda b,a:learned.compose_action(b,a,env._lowers,env._uppers,env.residual_bound))(
        jp.asarray(base_actions),jp.asarray(trace["action"]))
    full,requested = np.asarray(full),np.asarray(requested)
    np.testing.assert_allclose(requested[active],trace["requested_residual"][active],atol=1e-7,rtol=0)
    expected_desired = low+(full+1)*.5*(high-low)
    np.testing.assert_allclose(expected_desired[active],trace["desired_ctrl"][active],atol=2e-5,rtol=0)
    np.testing.assert_allclose(trace["effective_residual"][active],
        (trace["ctrl"]-trace["learned_base"])[active],atol=1e-7,rtol=0)
    # A separate algebraic check uses the actual saved base, independent of NN rounding.
    np.testing.assert_allclose(trace["desired_ctrl"][active],
        np.clip(trace["learned_base"]+requested,low,high)[active],atol=4e-7,rtol=0)
    reset = jax.jit(env.reset)
    recorded = {r["seed"]:r for r in json.loads((folder/"episodes.json").read_text())}
    rows = []
    for i,seed in enumerate(trace["seeds"]):
        state = jax.device_get(reset(jax.random.PRNGKey(int(seed))))
        np.testing.assert_allclose(trace["obs"][i,0],state.obs["state"],atol=1e-7,rtol=0)
        indices = np.flatnonzero(active[i])
        if len(indices)>1:
            np.testing.assert_allclose(trace["obs"][i,indices[1:],51:68],full[i,indices[:-1]],atol=1e-5,rtol=0)
        expected_slot = np.asarray(state.obs["state"][95:112])
        np.testing.assert_array_equal(trace["obs"][i,indices,95:112],np.broadcast_to(expected_slot,(len(indices),17)))
        sim = SuspendedCylinder(model.read_text())
        sim.reset(np.asarray(state.info["cube_spawn_pos"]),np.asarray(state.data.qpos[env.oid+3:env.oid+7]))
        mujoco.mj_setState(sim.model,sim.data,np.asarray(state.data.integration_words).copy().view(np.float64),STATE_SPEC)
        max_q = max_v = max_audit = 0.
        for k in indices:
            row = sim.step(trace["desired_ctrl"][i,k])
            max_q = max(max_q,float(np.max(np.abs(trace["qpos"][i,k]-sim.data.qpos))))
            max_v = max(max_v,float(np.max(np.abs(trace["qvel"][i,k]-sim.data.qvel))))
            packed = np.r_[row["loads"],row["forbidden"],row["penetration_m"],row["peak_force_n"],
                row["rope_force_n"],row["alive"],row["released"],row["valid"],row["time"]]
            max_audit = max(max_audit,float(np.max(np.abs(trace["audit"][i,k]-packed))))
            assert bool(trace["alive"][i,k]) == row["alive"]
            np.testing.assert_allclose(trace["hold"][i,k],row["hold_sec"] if row["alive"] else 0,atol=5e-6)
        assert recorded[int(seed)]["success30"] == row["success30"]
        assert max_q<3e-6 and max_v<1e-4 and max_audit<1e-4
        rows.append(dict(seed=int(seed),success30=row["success30"],executed_steps=len(indices),
            max_qpos_error=max_q,max_qvel_error=max_v,max_audit_error=max_audit))
    result = dict(status="verified",episodes=rows,trace_sha256=digest(folder/"preview_trajectories.npz"),
        checkpoint_sha256=digest(folder/"params.pkl"),base_sha256=digest(base_path),
        max_policy_action_error=float(np.max(np.abs(predicted[active]-trace["action"][active]))),
        max_base_control_error_rad=float(np.max(np.abs(expected_base[active]-trace["learned_base"][active]))),
        normalized_action_tolerance=1e-5,base_control_tolerance_rad=2e-5,
        composition_tolerance_rad=4e-7,full_command_history_verified=True,
        scope="All executed saved-preview frames, not unsaved evaluation trajectories",
        source_sha256=digest(Path(__file__)))
    (folder/"native_replay_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
