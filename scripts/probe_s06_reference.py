"""Native, substep-audited cylinder reference feasibility (not PPO)."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from diagnose_orca_grasp import model_indices, contact_state
from orca_cylinder_s02_env import cylinder_half_height
from orca_s06_env import cylinder_contacts
from probe_hic53_orca import cube_penetration


def rollout(model, packet, seed, seconds=40):
    qids, oid, _ = model_indices(model)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    initial, reference = packet["initial_ctrl"], packet["ctrl"]
    spawn = packet["object_position"] + rng.uniform(-1, 1, 3)*[.002, .002, 0]
    data.qpos[qids] = np.clip(initial+rng.normal(0, .002, 17), *model.actuator_ctrlrange.T)
    data.qpos[oid:oid+7] = np.r_[spawn, packet["object_quaternion"]]
    data.ctrl[:] = initial
    support = spawn-[0, 0, cylinder_half_height(packet["object_quaternion"])+.004]
    data.mocap_pos[0] = support
    mujoco.mj_forward(model, data)
    reset_depth = cube_penetration(model, data)
    records, hold, best = [], 0, 0
    for step in range(round(seconds/.05)):
        t = (step+1)*.05
        data.ctrl[:] = np.clip(reference[min(step, len(reference)-1)], data.ctrl-.1, data.ctrl+.1)
        data.mocap_pos[0] = support-[0, 0, np.clip((t-2)*.2, 0, .2)]
        valid, depth, forbidden, force = True, 0., False, 0.
        good_min, bad_max = np.full(3, np.inf), np.zeros(3)
        for _ in range(round(.05/model.opt.timestep)):
            mujoco.mj_step(model, data)
            good, bad, _ = cylinder_contacts(model, data)
            good_min, bad_max = np.minimum(good_min, good), np.maximum(bad_max, bad)
            _, bad_body, loads = contact_state(model, data)
            forbidden |= bool(bad_body)
            force = max(force, loads)
            depth = max(depth, cube_penetration(model, data))
            valid &= bool((good > .02).all() and not (bad > .02).any())
        alive = bool(np.isfinite(data.qpos).all() and data.qpos[oid+2] >= spawn[2]-.055)
        valid &= alive and t >= 3 and data.qpos[oid+2] > spawn[2]-.01 and not forbidden and depth <= .003
        hold = hold+1 if valid else 0
        best = max(best, hold)
        records.append(dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                            mocap=data.mocap_pos.copy(), good=good_min, bad=bad_max, hold=hold*.05,
                            alive=alive, forbidden=forbidden, penetration=depth, force=force))
        if not alive:
            break
    trace = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    report = dict(seed=seed, hold_sec=best*.05, final_valid=bool(valid), reset_depth_m=reset_depth,
                  success30=bool(best*.05 >= 30 and valid), duration_sec=len(records)*.05,
                  max_penetration_m=float(trace["penetration"].max()), peak_force_n=float(trace["force"].max()),
                  final_good=trace["good"][-1].tolist(), final_bad=trace["bad"][-1].tolist())
    return report, trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--seconds", type=float, default=40)
    p.add_argument("--disable-hand-contacts", action="store_true", help="Negative control only, never a candidate model")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(args.input / "model.xml"))
    if args.disable_hand_contacts:
        mask = model.geom_contype == 1
        model.geom_contype[mask] = 0
        model.geom_conaffinity[mask] = 0
        (args.output / "negative_control.json").write_text(json.dumps(dict(
            hand_contact_geoms_disabled=int(mask.sum()), model_equality_constraints=int(model.neq),
            note="Negative control: all physical hand contacts disabled, same object gravity, no policy training"), indent=2)+"\n")
    with np.load(args.input / "reference.npz") as z:
        packet = dict(z)
    rows = []
    for seed in range(6101, 6101+args.count):
        report, trace = rollout(model, packet, seed, args.seconds)
        rows.append(report)
        np.savez_compressed(args.output / f"trace_{seed}.npz", **trace)
        print(json.dumps(report), flush=True)
    (args.output / "report.json").write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    main()
