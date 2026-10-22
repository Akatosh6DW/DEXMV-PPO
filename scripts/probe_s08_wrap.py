"""Small native palm-wrap feasibility study, separate from frozen S07 and RL."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from diagnose_orca_grasp import FINGERS, model_indices
from orca_cylinder_s02_env import cylinder_half_height
from prepare_cylinder_s02_source import digest
from probe_hic53_orca import cube_penetration

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/orca_benchmark_suite/S07/inputs/model.xml"
OUT = ROOT / "artifacts/orca_benchmark_suite/S08_wrap_pilot"


def contacts(m, d):
    cube = m.geom("cube").id
    axis = d.geom_xmat[cube].reshape(3, 3)[:, 2]
    good = np.zeros(6)
    bad, peak = False, 0.
    for ci, c in enumerate(d.contact[:d.ncon]):
        if cube not in c.geom or c.dist > 0:
            continue
        gid = c.geom2 if c.geom1 == cube else c.geom1
        name = m.geom(gid).name or ""
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, ci, f)
        if f[0] <= 1e-5:
            continue
        peak = max(peak, float(f[0]))
        normal = c.frame[:3] * (1 if c.geom1 == gid else -1)
        local = normal @ d.xmat[m.geom_bodyid[gid]].reshape(3, 3)
        radial = abs(normal @ axis) <= .5
        if name.startswith("orca_palm_"):
            if radial and local[1] >= .6:
                good[0] += f[0]
        else:
            finger = next((i for i, n in enumerate(FINGERS) if name.startswith(f"orca_{n}_")), None)
            if finger is not None:
                if radial and ("_tip_" in name or "_prox_" in name) and -local[0] >= .4:
                    good[finger+1] += f[0]
            else:
                bad = True
    return good, bad, peak


def rollout(m, packet, seed, seconds, disable_contacts=False):
    qids, oid, _ = model_indices(m)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    spawn = packet["object_position"] + rng.uniform(-.002, .002, 3)*[1, 1, 0]
    d.qpos[qids] = np.clip(packet["initial_ctrl"]+rng.normal(0, .002, 17), *m.actuator_ctrlrange.T)
    d.qpos[oid:oid+7] = np.r_[spawn, packet["object_quaternion"]]
    support = spawn-[0, 0, cylinder_half_height(packet["object_quaternion"])+.004]
    d.mocap_pos[0] = support
    d.ctrl[:] = packet["initial_ctrl"]
    mujoco.mj_forward(m, d)
    reset_depth = cube_penetration(m, d)
    old = m.geom_contype.copy(), m.geom_conaffinity.copy()
    if disable_contacts:
        mask = m.geom_contype == 1
        m.geom_contype[mask], m.geom_conaffinity[mask] = 0, 0
    rows, hold, best, any_bad, max_depth = [], 0, 0, False, 0.
    for k in range(round(seconds/.05)):
        t = (k+1)*.05
        target = np.array([np.interp(t, packet["time"], packet["ctrl"][:, j]) for j in range(17)])
        d.ctrl[:] = np.clip(np.clip(target, *m.actuator_ctrlrange.T), d.ctrl-.1, d.ctrl+.1)
        d.mocap_pos[0] = support-[0, 0, np.clip((t-2)*.2, 0, .2)]
        loads, dirty, peak, depth = np.full(6, np.inf), False, 0., 0.
        for _ in range(round(.05/m.opt.timestep)):
            mujoco.mj_step(m, d)
            good, forbidden, force = contacts(m, d)
            loads = np.minimum(loads, good)
            dirty |= forbidden
            peak = max(peak, force)
            depth = max(depth, cube_penetration(m, d))
        finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        alive = finite and d.qpos[oid+2] >= spawn[2]-.055
        force_wrap = bool(loads[0] > .02 and loads[1] > .02 and np.sum(loads[2:] > .02) >= 2)
        valid = alive and t >= 3 and d.qpos[oid+2] > spawn[2]-.015 and force_wrap and not dirty and depth <= .003
        if t >= 3:
            any_bad |= dirty
            max_depth = max(max_depth, depth)
        hold = hold+1 if valid else 0
        best = max(best, hold)
        rows.append(dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), mocap=d.mocap_pos.copy(),
            loads=loads, hold=hold*.05, alive=alive, forbidden=dirty, penetration=depth, peak_force=peak,
            surface_groups=int(np.sum(loads > .02)), force_wrap=force_wrap))
        if not alive:
            break
    m.geom_contype[:], m.geom_conaffinity[:] = old
    trace = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    report = dict(seed=seed, hold_sec=best*.05, duration_sec=len(rows)*.05, final_valid=bool(valid),
        success30=bool(best*.05 >= 30 and valid and not any_bad and max_depth <= .003 and reset_depth <= .001),
        reset_penetration_m=reset_depth, max_released_penetration_m=max_depth, forbidden=bool(any_bad),
        peak_force_n=float(trace["peak_force"].max()), final_loads=loads.tolist(),
        max_surface_groups=int(trace["surface_groups"].max()),
        wrap_fraction=float(trace["force_wrap"][59:].mean()) if len(rows)>59 else 0.)
    return report, trace


def fit(m, direction, start):
    qids, oid, _ = model_indices(m)
    d = mujoco.MjData(m)
    gids = np.array([m.geom(f"orca_{f}_tip_proxy_fit_{'02' if f=='thumb' else '03'}").id for f in FINGERS])
    bodies = m.geom_bodyid[gids]
    radius = .0265+m.geom_size[gids, 0]-.0005
    # Body Y is the cylinder axis; the geom has a fixed X rotation.
    rot = Rotation.from_euler("x", -np.pi/2).as_matrix() if direction == "longitudinal" else Rotation.from_euler("z", -np.pi/2).as_matrix()
    quat = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
    low = np.r_[m.actuator_ctrlrange[1:, 0]+.005, [.068, .038, .225]]
    high = np.r_[m.actuator_ctrlrange[1:, 1]-.005, [.098, .065, .255]]
    center = np.array([.082, .048, .239])
    rng = np.random.default_rng(8101+start)
    x = np.r_[np.zeros(16), center]
    x[[1, 4, 7, 10, 14]] = .9
    x[[2, 5, 8, 11, 15]] = 1.2
    x += rng.normal(0, .18, 19)*np.r_[np.ones(16), np.full(3, .01)]
    x = np.clip(x, low+1e-6, high-1e-6)
    cube = m.geom("cube").id
    def objective(x):
        pos = x[16:]
        ctrl = np.r_[0., x[:16]]
        d.qpos[qids] = ctrl
        d.qpos[oid:oid+7] = np.r_[pos, quat]
        d.mocap_pos[0] = [0, 0, -1]
        mujoco.mj_forward(m, d)
        outward = np.zeros((5, 3))
        if direction == "longitudinal":
            theta = np.deg2rad([185, 120, 90, 65, 35])
            outward[:, :2] = np.c_[np.cos(theta), np.sin(theta)]
            target = pos+outward*radius[:, None]
            target[:, 2] += [-.013, .011, .013, .003, -.010]
        else:
            theta = np.deg2rad([-75, 20, 0, -15, -30])
            outward[:, 1:] = np.c_[np.cos(theta), np.sin(theta)]
            target = pos+outward*radius[:, None]
            target[:, 0] += [-.017, -.019, -.002, .015, .021]
        normal = -d.xmat[bodies].reshape(5, 3, 3)[:, :, 0]
        penalties = np.zeros(m.ngeom)
        for c in d.contact[:d.ncon]:
            if cube in c.geom:
                other = c.geom2 if c.geom1 == cube else c.geom1
                if m.geom_contype[other] == 1:
                    penalties[other] = max(penalties[other], -c.dist-.001)
        return np.r_[(d.geom_xpos[gids]-target).ravel(), (.008*(normal+outward)).ravel(),
                     5*penalties, .4*(pos-center), .0001*x[:16]]
    result = least_squares(objective, x, bounds=(low, high), max_nfev=160)
    ctrl = np.r_[0., result.x[:16]]
    times = np.arange(31)*.05
    packet = dict(time=times, ctrl=(times/1.5)[:, None]*ctrl, initial_ctrl=np.zeros(17),
                  object_position=result.x[16:], object_quaternion=quat,
                  object_extents_m=np.full(3, .053))
    return packet, float(np.linalg.norm(objective(result.x)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(BASE))
    manifest = dict(base=str(BASE), base_sha256=digest(BASE), physics_unchanged=True, equality_constraints=int(m.neq),
        scope="Native geometry/reference feasibility only, not BC/PPO, no new video extraction. Manual wrap targets, no claims of video attribution.",
        criterion="Palm curved-side load >.02 N AND thumb >.02 N AND at least two opposing fingers >.02 N, every5ms; after3s, height loss<15mm, penetration<=3mm, no support/floor contact; continuous30s and valid at40s.",
        development_seed=8101, test_seeds=list(range(81200, 81208)), starts_per_direction=args.starts)
    (OUT / "protocol.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for direction in ("longitudinal", "transverse"):
        target = OUT / direction
        target.mkdir()
        rows, candidates = [], []
        for start in range(args.starts):
            packet, error = fit(m, direction, start)
            report, trace = rollout(m, packet, 8101, 8.)
            score = report["hold_sec"] + report["wrap_fraction"] - 2000*max(report["reset_penetration_m"]-.001, 0) - 2000*max(report["max_released_penetration_m"]-.003, 0)
            rows.append(dict(start=start, fit_error=error, score=score, **report))
            candidates.append(packet)
            np.savez_compressed(target / f"candidate_{start}.npz", **packet)
            print("DEVELOPMENT", direction, json.dumps(rows[-1]), flush=True)
        best = max(range(len(rows)), key=lambda i: (rows[i]["score"], -rows[i]["fit_error"]))
        packet = candidates[best]
        np.savez_compressed(target / "reference.npz", **packet)
        tree = ET.parse(BASE)
        body = tree.find(".//body[@name='cube']")
        body.set("pos", " ".join(map(str, packet["object_position"])))
        body.set("quat", " ".join(map(str, packet["object_quaternion"])))
        support = packet["object_position"]-[0, 0, cylinder_half_height(packet["object_quaternion"])+.004]
        tree.find(".//body[@name='cube_support']").set("pos", " ".join(map(str, support)))
        tree.write(target / "model.xml", encoding="unicode")
        # Freeze before independent validation; unsuccessful candidates remain.
        (target / "selection.json").write_text(json.dumps(dict(best_start=best, development=rows, reference_sha256=digest(target / "reference.npz")), indent=2)+"\n")
        reports = []
        for seed in range(81200, 81208):
            report, trace = rollout(m, packet, seed, 40.)
            reports.append(report)
            np.savez_compressed(target / f"trace_{seed}.npz", **trace)
            print("TEST", direction, json.dumps(report), flush=True)
        negative, trace = rollout(m, packet, 81200, 8., disable_contacts=True)
        np.savez_compressed(target / "negative_trace.npz", **trace)
        (target / "report.json").write_text(json.dumps(dict(selected_start=best, tests=reports,
            successes=sum(r["success30"] for r in reports), count=len(reports), negative_control=negative), indent=2)+"\n")


if __name__ == "__main__":
    main()
