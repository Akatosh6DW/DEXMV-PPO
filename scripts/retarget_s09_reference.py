"""Retarget processed DexMV cylinder samples at prescribed, unfiltered orientations.

Cylinder axial symmetry allows choosing the grasp azimuth without changing the
object pose. This is bounded geometric retargeting, not new video inference.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from s09_suspension import (OUT, SOURCE, SuspendedCylinder, body_quaternion,
                            orientation_samples, rollout, scene_xml)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Retargeter:
    def __init__(self, xml=None):
        self.sim = SuspendedCylinder(xml)
        self.model, self.data = self.sim.model, self.sim.data
        self.chosen = np.r_[0, np.arange(7, 17)]
        self.low, self.high = self.model.actuator_ctrlrange[self.chosen].T
        self.pads = np.array([self.model.geom(f"orca_{f}_tip_proxy_fit_{n}").id
                             for f, n in (("thumb", "02"), ("index", "03"), ("middle", "03"))])
        self.bodies = self.model.geom_bodyid[self.pads]
        with np.load(SOURCE) as z:
            self.source = z["fingertips_centered"][:, :3] * (.053/z["source_extents_m"])
            self.source_time = z["time"] - z["time"][0]
        radial = np.linalg.norm(self.source[:, :, :2], axis=-1)
        self.outward = np.zeros_like(self.source)
        self.outward[:, :, :2] = self.source[:, :, :2] / np.maximum(radial[:, :, None], 1e-9)
        self.near = np.clip((.02-np.abs(radial-.0265))/.012, 0, 1)
        self.targets = self.source.copy()
        radius = .0265+self.model.geom_size[self.pads, 0]-.0006
        self.targets[:, :, :2] += self.near[:, :, None]*(self.outward[:, :, :2]*radius[None, :, None]-self.source[:, :, :2])

    def fit(self, position, quaternion, *, starts=6, seed=9011, max_nfev=180):
        m, d, sim = self.model, self.data, self.sim
        sim.reset(position, quaternion)
        initial_depth = sim.initial_penetration
        geom_rotation = d.geom_xmat[m.geom("cube").id].reshape(3, 3).copy()
        low = np.r_[self.low+.002, -np.pi]
        high = np.r_[self.high-.002, np.pi]
        n = len(self.chosen)

        def residual(x, frame, regularize=None):
            q, angle = x[:n], x[-1]
            d.qpos[sim.qids] = 0
            d.qpos[sim.qids[self.chosen]] = q
            mujoco.mj_forward(m, d)
            rotation = geom_rotation @ Rotation.from_euler("z", angle).as_matrix()
            normal = -d.xmat[self.bodies].reshape(3, 3, 3)[:, :, 0]
            errors = d.geom_xpos[self.pads]-(self.targets[frame]@rotation.T+position)
            penalties = np.zeros(m.ngeom)
            cube = m.geom("cube").id
            for c in d.contact[:d.ncon]:
                if cube not in c.geom:
                    continue
                other = c.geom2 if c.geom1 == cube else c.geom1
                if m.geom_contype[other] == 1:
                    penalties[other] = max(penalties[other], -c.dist-.0008)
            out = np.r_[errors.ravel(),
                        (.012*self.near[frame, :, None]*(normal+self.outward[frame]@rotation.T)).ravel(),
                        6*penalties]
            return out if regularize is None else np.r_[out, .0005*(q-regularize)]

        rng = np.random.default_rng(seed)
        candidates = []
        for start in range(starts):
            x = rng.uniform(low, high)
            result = least_squares(lambda v: residual(v, -1), x, bounds=(low, high), max_nfev=max_nfev)
            candidates.append(dict(start=start, cost=float(np.linalg.norm(residual(result.x, -1))),
                                   x=result.x.tolist(), nfev=result.nfev))
        selected = min(candidates, key=lambda r: r["cost"])
        x = np.asarray(selected["x"])
        raw_frames = np.unique(np.linspace(0, len(self.source)-1, 16).astype(int))
        controls = np.zeros((len(raw_frames), m.nu))
        errors = []
        for i in reversed(range(len(raw_frames))):
            angle, previous = x[-1], x[:n].copy()
            result = least_squares(lambda v: residual(np.r_[v, angle], raw_frames[i], previous),
                                   previous, bounds=(low[:n], high[:n]), max_nfev=100)
            x = np.r_[result.x, angle]
            controls[i, self.chosen] = result.x
            errors.append(float(np.linalg.norm(residual(x, raw_frames[i])[:9])))
        time = np.arange(41)*.05
        fraction = np.minimum(time/2., 1)
        source_t = self.source_time[raw_frames]/self.source_time[-1]
        ctrl = np.stack([np.interp(fraction, source_t, controls[:, j]) for j in range(m.nu)], axis=1)
        ctrl *= fraction[:, None]
        packet = dict(time=time, ctrl=ctrl, initial_ctrl=np.zeros(m.nu),
                      object_position=np.asarray(position), object_quaternion=np.asarray(quaternion),
                      source_frames=raw_frames, raw_ik_ctrl=controls, source_local_tips=self.source,
                      source_targets=self.targets, axial_azimuth=np.asarray(selected["x"][-1]),
                      object_extents_m=np.full(3, .053))
        report = dict(candidates=candidates, selected_start=selected["start"],
                      endpoint_cost=selected["cost"], sample_tip_error_m=errors[::-1],
                      initial_penetration_m=float(initial_depth), source_sha256=digest(SOURCE),
                      source=str(SOURCE), source_sequence=96, source_frames=[182, 271],
                      fingers=["thumb", "index", "middle"],
                      exclusions="Ring is distant; little finger source interpenetration is unreliable.",
                      reference_origin="Processed DexMV relative fingertips + constrained IK, no new RGB extraction")
        return packet, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--random-count", type=int, default=3)
    p.add_argument("--starts", type=int, default=6)
    p.add_argument("--seed", type=int, default=9011)
    p.add_argument("--seconds", type=float, default=8.)
    p.add_argument("--position", type=float, nargs=3, default=[.08, .055, .25])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    xml = scene_xml()
    (args.output/"model.xml").write_text(xml)
    retargeter = Retargeter(xml)
    # All requested poses remain in the report, even when IK or physics fails.
    quats = orientation_samples(args.random_count, args.seed, include_axes=True)
    rows = []
    for i, q in enumerate(quats):
        folder = args.output/f"pose_{i:03d}"
        folder.mkdir()
        packet, fit = retargeter.fit(args.position, q, starts=args.starts, seed=args.seed+i)
        np.savez_compressed(folder/"reference.npz", **packet)
        (folder/"retarget.json").write_text(json.dumps(fit, indent=2)+"\n")
        report, trace = rollout(packet, args.seed, args.seconds, xml)
        np.savez_compressed(folder/"trace.npz", **trace)
        row = dict(pose=i, quaternion=q.tolist(), endpoint_cost=fit["endpoint_cost"], **report)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (args.output/"development.json").write_text(json.dumps(rows, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(dict(status="development_only", source_sha256=digest(SOURCE),
        model_sha256=digest(args.output/"model.xml"), orientations=len(quats), seed=args.seed,
        position=args.position, starts=args.starts, seconds=args.seconds,
        final_test_used=False), indent=2)+"\n")


if __name__ == "__main__":
    main()
