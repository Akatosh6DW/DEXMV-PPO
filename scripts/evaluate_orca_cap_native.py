#!/usr/bin/env python3
"""Independent native MuJoCo policy rollout, not a saved-state video replay."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

from brax.training.agents.ppo import networks
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from orca_cap_scene import contacts, indices, PITCH
from orca_cap_metrics import hold_seconds
from train_orca_comparison import network_factory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--count", type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.policy.open("rb") as f:
        params = pickle.load(f)
    net = network_factory({"state": (95,), "privileged_state": (112,)}, 17)
    policy = jax.jit(networks.make_inference_fn(net)(params, deterministic=True))
    model = mujoco.MjModel.from_xml_path(str(args.model))
    qids, turn, lift, tips = indices(model)
    dqids = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    cap = model.geom("cap").id
    bid = model.body("cap").id
    center = model.body("fixed_bottle").pos.copy()
    low, high = model.actuator_ctrlrange.T

    def normalize(ctrl):
        return np.clip(2*(ctrl-low)/(high-low)-1, -1, 1)

    episodes, records = [], []
    for seed in range(args.start, args.start + args.count):
        d = mujoco.MjData(model)
        initial = np.zeros(17)
        initial[0] = -.25
        key, qkey = jax.random.split(jax.random.PRNGKey(seed))
        d.qpos[qids] = np.clip(initial + .002*np.asarray(jax.random.normal(qkey, (17,))), low, high)
        d.ctrl[:] = initial
        mujoco.mj_forward(model, d)
        last = normalize(initial)
        alive, invalid = True, False
        count = best = 0
        rows = []
        for t in range(240):
            pos = d.geom_xpos[cap]
            obs = np.concatenate([normalize(d.qpos[qids]), np.clip(d.qvel[dqids]*.1, -5, 5),
                                  normalize(d.ctrl), last, (pos-center)*20, d.cvel[bid, 3:]*2,
                                  d.xquat[bid], ((d.geom_xpos[tips]-pos)*10).ravel(),
                                  [min(t*.05/3, 1), d.qpos[turn]/(np.pi/2)]]).astype(np.float32)
            action = np.asarray(policy({"state": jp.array(obs)}, key)[0])
            target = low + .5*(np.clip(action, -1, 1)+1)*(high-low)
            d.ctrl[:] = np.clip(target, d.ctrl-.1, d.ctrl+.1)
            mujoco.mj_step(model, d, nstep=round(.05/model.opt.timestep))
            finger, forbidden, force = contacts(model, d)
            angle, axial = d.qpos[turn], d.qpos[lift]
            error = abs(axial-PITCH*angle)
            finite = np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
            alive &= finite and error <= .003
            invalid |= forbidden or error > .001 or not finite
            paired = bool(finger[0] and finger[1:].any())
            valid = alive and not invalid and paired and angle >= np.pi/2 and error < .0002
            count = count+1 if valid else 0
            best = max(best, count)
            rows.append(dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), action=action,
                             angle=angle, degrees=np.rad2deg(angle), lift=axial, thread_error=error,
                             paired=paired, forbidden=forbidden, force=force, alive=alive,
                             hold_sec=count*.05))
            last = action
        episodes.append({k: np.asarray([r[k] for r in rows]) for k in rows[0]})
        records.append(dict(seed=seed, best_hold_sec=best*.05, success=best >= 20,
                            final_degrees=float(np.rad2deg(d.qpos[turn]))))
    trace = {k: np.stack([e[k] for e in episodes]) for k in episodes[0]}
    np.savez_compressed(args.output / "trajectories.npz", **trace)
    audited_holds = hold_seconds(trace)
    for record, hold in zip(records, audited_holds):
        record.update(best_hold_sec=float(hold), success=bool(hold >= 1.))
    report = dict(backend="native_mujoco_fresh_policy_rollouts", seeds=list(range(args.start, args.start+args.count)),
                  policy=str(args.policy.resolve()), policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  model=str(args.model.resolve()), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  success_rate=float(np.mean([r["success"] for r in records])),
                  mean_hold_sec=float(np.mean([r["best_hold_sec"] for r in records])),
                  max_thread_error_m=float(trace["thread_error"].max()),
                  forbidden_episode_fraction=float(trace["forbidden"].any(axis=1).mean()),
                  peak_force_n=float(trace["force"].max()), records=records)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
