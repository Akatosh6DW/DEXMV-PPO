"""Replay only training demonstrations to obtain physically integrated RSI states.

This prepares training reset states, not PPO transitions or test successes.
Recorded normalized actions are replayed through the current full-target
interface; this is not claimed to bitwise reproduce historical collection.
"""

import argparse
import copy
import json
from pathlib import Path
import pickle
import shutil

import jax
import jax.numpy as jp
import mujoco
import numpy as np

import run_s09_job as runner
from run_s09_bc_conditioned import ModeConditionedGrasp
from retarget_s09_reference import digest
from s09_suspension import SuspendedCylinder
from probe_s09_recovery_teacher import SPEC


FRAMES = (40, 80)


def eligible(state, frame):
    clean = (float(state.info["initial_penetration"]) <= .001
             and float(state.info["max_depth"]) <= .003
             and not bool(state.info["ever_forbidden"]) and not bool(state.done))
    return clean and (frame < 60 or
        (float(state.metrics["stable"]) > 0 and float(state.data.audit[8]) == 0))


def clear_credit(state):
    # Restore dynamics and phase, but do not inherit the teacher's held duration.
    info = dict(state.info, hold_steps=np.int32(0), best_steps=np.int32(0))
    metrics = {key:np.float32(0) for key in state.metrics}
    return state.replace(info=info, metrics=metrics, reward=np.float32(0), done=np.float32(0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    allowed = runner.ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
    if not args.output.resolve().is_relative_to(allowed):
        p.error("Use the separate improvement artifacts root")
    jax.config.update("jax_default_matmul_precision","highest")
    source = runner.ROOT/"artifacts/orca_benchmark_suite/S09/demonstrations"
    model = source.parent/"inputs/model.xml"
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    assert manifest["dataset_sha256"] == digest(source/"trajectories.npz")
    assert manifest["model_sha256"] == digest(model)
    assert manifest["reference_sha256"] == digest(model.parent/"reference.npz")
    split = manifest.get("train_episodes",len(manifest["seeds"])*3//4)
    count = 2 if args.smoke else split
    seeds = np.asarray(manifest["seeds"][:count],np.int32)
    with np.load(source/"trajectories.npz",allow_pickle=False) as z:
        actions = z["action"][:count,:max(FRAMES)].copy()
        assert z["active"][:count,:max(FRAMES)].all()
    env = ModeConditionedGrasp(model,"pure")

    @jax.jit
    def collect(seed, actions):
        initial = env.reset(jax.random.PRNGKey(seed))
        def step(state, action):
            nxt = env.step(state, action)
            return nxt, (nxt,nxt.info["desired_ctrl"])
        _,(states,desired) = jax.lax.scan(step,initial,actions)
        selected = jax.tree_util.tree_map(lambda x:x[jp.asarray(FRAMES)-1],states)
        ever_done = jp.cumsum(states.done)>0
        return initial, selected, desired, ever_done[jp.asarray(FRAMES)-1]

    args.output.mkdir(parents=True,exist_ok=False)
    protocol = dict(status="running", source_dataset_sha256=manifest["dataset_sha256"],
        source_manifest_sha256=digest(source/"dataset_manifest.json"), model_sha256=digest(model),
        reference_sha256=manifest["reference_sha256"], source_sha256=digest(Path(__file__)),
        source_training_split=split, seeds=seeds.tolist(), frames=list(FRAMES), backend=jax.default_backend(),
        selection="All first training-split episodes; fixed frames 2s and 4s; reject invalid starts",
        hold_credit_reset=True, validation_or_test_used=False, new_human_videos=0,
        limitation="Changes training initial-state distribution only; no benchmark success claim")
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    shutil.copyfile(Path(__file__),args.output/Path(__file__).name)
    states, records = [], []
    max_q = max_v = 0.
    for seed, commands in zip(seeds,actions):
        initial, selected, desired, ever_done = jax.device_get(collect(jp.int32(seed),jp.asarray(commands)))
        sim = SuspendedCylinder(model.read_text())
        sim.reset(np.asarray(initial.info["cube_spawn_pos"]),
                  np.asarray(initial.data.qpos[sim.oid+3:sim.oid+7]))
        mujoco.mj_setState(sim.model,sim.data,np.asarray(initial.data.integration_words).copy().view(np.float64),SPEC)
        candidates = {frame:jax.tree_util.tree_map(lambda x:x[i],selected) for i,frame in enumerate(FRAMES)}
        for frame, ctrl in enumerate(desired,1):
            row = sim.step(ctrl)
            if frame not in candidates:
                continue
            state = candidates[frame]
            q_error = float(np.max(np.abs(sim.data.qpos-state.data.qpos)))
            v_error = float(np.max(np.abs(sim.data.qvel-state.data.qvel)))
            max_q, max_v = max(max_q,q_error), max(max_v,v_error)
            assert q_error < 3e-6 and v_error < 1e-4
            failed_prefix = bool(ever_done[FRAMES.index(frame)])
            accepted = eligible(state,frame) and not failed_prefix
            record = dict(seed=int(seed),frame=frame,accepted=accepted,
                valid=bool(state.metrics["stable"]),alive=not bool(state.done),
                failed_earlier_in_prefix=failed_prefix,
                released_rope_force_n=float(row["rope_force_n"]),
                max_released_penetration_m=float(state.info["max_depth"]),
                qpos_error=q_error,qvel_error=v_error)
            if accepted:
                states.append(clear_credit(copy.deepcopy(state)))
            records.append(record)
        print(json.dumps(dict(seed=int(seed),accepted=sum(r["accepted"] for r in records if r["seed"]==seed))),flush=True)
    if not states:
        raise ValueError("No physically qualified reset states")
    bank = jax.tree_util.tree_map(lambda *xs:np.stack(xs),*states)
    path = args.output/"states.pkl"
    path.write_bytes(pickle.dumps(bank))
    # Integration words include velocities, warm starts, simulation time and ctrl.
    protocol.update(status="completed",accepted=len(states),attempted=len(records),records=records,
        state_sha256=digest(path),collection_interactions=len(seeds)*max(FRAMES),
        native_validation_interactions=len(seeds)*max(FRAMES),
        pose_counts=np.bincount(bank.info["reference_index"],minlength=6).tolist(),
        max_qpos_error=max_q,max_qvel_error=max_v,policy_updates=0)
    (args.output/"manifest.json").write_text(json.dumps(protocol,indent=2)+"\n")
    print(json.dumps({k:v for k,v in protocol.items() if k!="records"}),flush=True)


if __name__ == "__main__":
    main()
