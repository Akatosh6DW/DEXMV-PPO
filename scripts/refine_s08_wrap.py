"""Second wrap probe: collision-free spawn and explicit thumb pad orientation."""

import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from diagnose_orca_grasp import model_indices
from prepare_cylinder_s02_source import digest
from probe_s08_wrap import BASE, OUT as FIRST, rollout

OUT = FIRST.parent / "S08_wrap_refined"


def fit_thumb(m, packet, start):
    qids, oid, _ = model_indices(m)
    d = mujoco.MjData(m)
    ctrl = packet["ctrl"][-1].copy()
    pos, quat = packet["object_position"], packet["object_quaternion"]
    d.qpos[qids] = ctrl
    d.qpos[oid:oid+7] = np.r_[pos, quat]
    d.mocap_pos[0] = [0, 0, -1]
    mujoco.mj_forward(m, d)
    cube, pad = m.geom("cube").id, m.geom("orca_thumb_tip_proxy_fit_02").id
    axis = d.geom_xmat[cube].reshape(3, 3)[:, 2].copy()
    body = m.geom_bodyid[pad]
    radius = .0265+m.geom_size[pad, 0]-.0008
    lo, hi = m.actuator_ctrlrange[13:].T
    rng = np.random.default_rng(8201+start)
    def residual(x):
        d.qpos[qids[13:]] = x
        mujoco.mj_forward(m, d)
        delta = d.geom_xpos[pad]-pos
        axial = delta @ axis
        radial = delta-axial*axis
        distance = np.linalg.norm(radial)
        outward = radial/max(distance, 1e-9)
        normal = -d.xmat[body].reshape(3, 3)[:, 0]
        penalties = np.zeros(m.ngeom)
        for c in d.contact[:d.ncon]:
            if cube in c.geom:
                gid = c.geom2 if c.geom1 == cube else c.geom1
                if (m.geom(gid).name or "").startswith("orca_thumb_"):
                    penalties[gid] = max(penalties[gid], -c.dist-.001)
        return np.r_[distance-radius, 2*max(abs(axial)-.015, 0), .035*(normal+outward), 5*penalties]
    fit = least_squares(residual, rng.uniform(lo+.002, hi-.002), bounds=(lo+.001, hi-.001), max_nfev=300)
    ctrl[13:] = fit.x
    packet = dict(packet)
    packet["ctrl"] = (packet["time"]/1.5)[:, None]*ctrl
    # Preserve the optimized final grasp workspace but start outside the palm.
    # No object servo: the fingers must close the gap through real contacts.
    packet["object_position"] = pos + [0, .006, 0]
    return packet, float(np.linalg.norm(residual(fit.x)))


def main():
    OUT.mkdir(exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(BASE))
    (OUT / "protocol.json").write_text(json.dumps(dict(
        scope="Geometry/reset correction only; no RL or video extraction. S07 unchanged.",
        changes="Initial object 6mm farther from palm; thumb IK jointly targets cylindrical side distance and palmar-facing normal, with fixed other fingers.",
        base_sha256=digest(BASE), source_sha256=digest(__file__), starts=8,
        development_seed=8201, test_seeds=list(range(82200,82208))), indent=2)+"\n")
    for direction in ("longitudinal", "transverse"):
        directory = OUT / direction
        directory.mkdir()
        with np.load(FIRST / direction / "reference.npz") as z:
            original = dict(z)
        rows, candidates = [], []
        for start in range(8):
            packet, error = fit_thumb(m, original, start)
            result, trace = rollout(m, packet, 8201, 8.)
            score = result["hold_sec"] + result["wrap_fraction"] - 2000*max(result["reset_penetration_m"]-.001, 0) - 2000*max(result["max_released_penetration_m"]-.003, 0)
            rows.append(dict(start=start, fit_error=error, score=score, **result))
            candidates.append(packet)
            np.savez_compressed(directory / f"candidate_{start}.npz", **packet)
            print("DEVELOPMENT", direction, json.dumps(rows[-1]), flush=True)
        best = max(range(8), key=lambda i: (rows[i]["score"], -rows[i]["fit_error"]))
        packet = candidates[best]
        np.savez_compressed(directory / "reference.npz", **packet)
        shutil.copy2(FIRST / direction / "model.xml", directory / "model.xml")
        (directory / "selection.json").write_text(json.dumps(dict(best_start=best, development=rows,
            reference_sha256=digest(directory / "reference.npz")), indent=2)+"\n")
        rows = []
        for seed in range(82200,82208):
            result, trace = rollout(m, packet, seed, 40.)
            rows.append(result)
            np.savez_compressed(directory / f"trace_{seed}.npz", **trace)
            print("TEST", direction, json.dumps(result), flush=True)
        negative, trace = rollout(m, packet, 82200, 8., disable_contacts=True)
        np.savez_compressed(directory / "negative_trace.npz", **trace)
        (directory / "report.json").write_text(json.dumps(dict(selected_start=best, tests=rows,
            successes=sum(r["success30"] for r in rows), count=len(rows), negative_control=negative), indent=2)+"\n")


if __name__ == "__main__":
    main()
