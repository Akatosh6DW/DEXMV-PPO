"""Native S09 cylinder physics: unilateral suspension, released grasp audit.

The string acts at the center of mass (an idealized balanced sling), not an
orientation servo. All orientations remain free rotational degrees of freedom.
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from diagnose_orca_grasp import FINGERS, model_indices
from probe_hic53_orca import cube_penetration

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/orca_benchmark_suite/S07/inputs/model.xml"
SOURCE = ROOT / "artifacts/orca_cylinder_s02/source/selection96/selected_segment.npz"
OUT = ROOT / "artifacts/orca_benchmark_suite/S09"


@dataclass(frozen=True)
class Suspension:
    length: float = .20
    stiffness: float = 100.
    damping: float = 1.
    release_sec: float = 3.
    control_dt: float = .05
    episode_sec: float = 40.
    hold_sec: float = 30.
    max_ctrl_delta: float = .1


def rope_force(position, velocity, anchor, time, cfg=Suspension()):
    """Kelvin-Voigt tensile string: slack and released strings exert zero force."""
    delta = np.asarray(anchor) - position
    distance = float(np.linalg.norm(delta))
    if time >= cfg.release_sec - 1e-9 or distance <= cfg.length:
        return np.zeros(3)
    direction = delta / distance
    extension_rate = -float(np.dot(velocity, direction))
    tension = max(0., cfg.stiffness * (distance - cfg.length) + cfg.damping * extension_rate)
    return tension * direction


def scene_xml(base=BASE):
    tree = ET.parse(base)
    root = tree.getroot()
    root.set("model", "S09 free-orientation suspended cylinder")
    support = root.find(".//body[@name='cube_support']")
    support.set("pos", "0.08 0.07 0.45")
    for geom in support.findall("geom"):
        support.remove(geom)
    ET.SubElement(support, "site", name="s09_anchor", size=".003", rgba=".2 .2 .2 1")
    cube = root.find(".//body[@name='cube']")
    ET.SubElement(cube, "site", name="s09_sling", pos="0 0 0", size=".001", rgba=".7 .7 .7 1")
    tendon = ET.SubElement(root, "tendon")
    cable = ET.SubElement(tendon, "spatial", name="s09_rope_visual", width=".001",
                          limited="false", stiffness="0", damping="0", rgba=".7 .7 .7 1")
    ET.SubElement(cable, "site", site="s09_anchor")
    ET.SubElement(cable, "site", site="s09_sling")
    return ET.tostring(root, encoding="unicode")


def body_quaternion(axis, roll=0.):
    """The existing cylinder geom's local Z axis is body -Y, not body Z."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    helper = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([1., 0., 0.])
    x = np.cross(helper, axis)
    x /= np.linalg.norm(x)
    rotation = np.column_stack((x, -axis, np.cross(x, -axis)))
    rotation = Rotation.from_rotvec(axis * roll).as_matrix() @ rotation
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def orientation_samples(count, seed, include_axes=False):
    random = Rotation.random(count, random_state=np.random.default_rng(seed)).as_quat()[:, [3, 0, 1, 2]]
    if include_axes:
        random = np.concatenate(([body_quaternion(a) for a in np.eye(3)], random))
    return random


def contact_audit(model, data):
    """Require loaded palmar finger surfaces; the palm may assist, not replace them."""
    cube = model.geom("cube").id
    loads = np.zeros(5)
    forbidden, peak = False, 0.
    for i, c in enumerate(data.contact[:data.ncon]):
        if cube not in c.geom or c.dist > 0:
            continue
        gid = c.geom2 if c.geom1 == cube else c.geom1
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        peak = max(peak, float(force[0]))
        if force[0] <= 1e-5:
            continue
        name = model.geom(gid).name or ""
        finger = next((j for j, f in enumerate(FINGERS) if name.startswith(f"orca_{f}_")), None)
        if finger is not None:
            normal = c.frame[:3] * (1 if c.geom1 == gid else -1)
            local = normal @ data.xmat[model.geom_bodyid[gid]].reshape(3, 3)
            if ("_tip_" in name or "_prox_" in name) and -local[0] >= .4:
                loads[finger] += force[0]
        elif not name.startswith("orca_palm_"):
            forbidden = True
    return loads, forbidden, peak


class SuspendedCylinder:
    """One fully explicit native episode, shared by reference search and RL."""

    def __init__(self, xml=None, cfg=Suspension()):
        self.model = mujoco.MjModel.from_xml_string(scene_xml() if xml is None else xml)
        self.cfg = cfg
        self.data = mujoco.MjData(self.model)
        self.qids, self.oid, _ = model_indices(self.model)
        self.body = self.model.body("cube").id
        self.dof = self.model.jnt_dofadr[self.model.joint("cube_freejoint").id]
        self.nsub = round(cfg.control_dt / self.model.opt.timestep)
        if not np.isclose(self.nsub * self.model.opt.timestep, cfg.control_dt):
            raise ValueError("Control period must be divisible by the physics timestep")
        np.testing.assert_allclose(self.model.body_ipos[self.body], 0, atol=1e-10)

    def reset(self, position, quaternion, initial_ctrl=None, velocity=None):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        initial = np.zeros(m.nu) if initial_ctrl is None else np.asarray(initial_ctrl)
        d.qpos[self.qids], d.ctrl[:] = initial, initial
        d.qpos[self.oid:self.oid+7] = np.r_[position, quaternion]
        if velocity is not None:
            d.qvel[self.dof:self.dof+3] = velocity
        self.spawn = np.array(position, float)
        preload = m.body_mass[self.body] * -m.opt.gravity[2] / self.cfg.stiffness
        d.mocap_pos[0] = self.spawn + [0, 0, self.cfg.length + preload]
        mujoco.mj_forward(m, d)
        self.initial_penetration = cube_penetration(m, d)
        self.hold = self.best = 0
        self.ever_forbidden = False
        self.max_depth = 0.
        return self.initial_penetration

    def step(self, desired):
        m, d, cfg = self.model, self.data, self.cfg
        desired = np.clip(desired, *m.actuator_ctrlrange.T)
        d.ctrl[:] = np.clip(desired, d.ctrl-cfg.max_ctrl_delta, d.ctrl+cfg.max_ctrl_delta)
        loads = np.full(5, np.inf)
        forbidden, depth, peak, rope_peak = False, 0., 0., 0.
        first_time = d.time
        for _ in range(self.nsub):
            # qpos position is the COM by construction; no artificial torque or pose tracking.
            force = rope_force(d.qpos[self.oid:self.oid+3], d.qvel[self.dof:self.dof+3],
                               d.mocap_pos[0], d.time, cfg)
            d.xfrc_applied[:] = 0
            d.xfrc_applied[self.body, :3] = force
            rope_peak = max(rope_peak, float(np.linalg.norm(force)))
            mujoco.mj_step(m, d)
            good, bad, current_peak = contact_audit(m, d)
            loads = np.minimum(loads, good)
            forbidden |= bad
            depth = max(depth, cube_penetration(m, d))
            peak = max(peak, current_peak)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or np.any(d.warning.number):
            raise FloatingPointError("S09 nonfinite state or MuJoCo warning")
        released = first_time >= cfg.release_sec - 1e-9
        loss = self.spawn[2] - d.qpos[self.oid+2]
        alive = loss < .055 and np.linalg.norm(d.qpos[self.oid:self.oid+3]-self.spawn) < .12
        paired = loads[0] > .02 and np.any(loads[1:] > .02)
        valid = bool(released and alive and loss < .015 and paired and not forbidden
                     and depth <= .003 and self.initial_penetration <= .001 and rope_peak == 0)
        self.hold = self.hold + 1 if valid else 0
        self.best = max(self.hold, self.best)
        if released:
            self.ever_forbidden |= forbidden
            self.max_depth = max(self.max_depth, depth)
        return dict(time=float(d.time), alive=bool(alive), released=bool(released), valid=valid,
                    hold_sec=self.hold*cfg.control_dt, best_hold_sec=self.best*cfg.control_dt,
                    loads=loads, forbidden=bool(forbidden), penetration_m=depth, peak_force_n=peak,
                    rope_force_n=rope_peak, height_loss_m=float(loss), paired=bool(paired),
                    success30=bool(self.best*cfg.control_dt >= cfg.hold_sec and valid
                                   and not self.ever_forbidden and self.max_depth <= .003))


def rollout(packet, seed=0, seconds=40., xml=None, disable_hand=False, jitter=0.):
    sim = SuspendedCylinder(xml)
    if disable_hand:
        hand = sim.model.geom_contype == 1
        sim.model.geom_contype[hand] = sim.model.geom_conaffinity[hand] = 0
    rng = np.random.default_rng(seed)
    position = packet["object_position"] + rng.uniform(-jitter, jitter, 3)
    sim.reset(position, packet["object_quaternion"], packet["initial_ctrl"])
    rows = []
    for _ in range(round(seconds/sim.cfg.control_dt)):
        t = sim.data.time + sim.cfg.control_dt
        ctrl = np.array([np.interp(t, packet["time"], packet["ctrl"][:, j]) for j in range(sim.model.nu)])
        row = sim.step(ctrl)
        row.update(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy(), ctrl=sim.data.ctrl.copy(),
                   mocap=sim.data.mocap_pos.copy())
        rows.append(row)
        if not row["alive"]:
            break
    trace = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    report = dict(seed=seed, duration_sec=float(trace["time"][-1]),
                  initial_penetration_m=float(sim.initial_penetration),
                  best_hold_sec=float(trace["best_hold_sec"][-1]),
                  success30=bool(trace["success30"][-1]), final_valid=bool(trace["valid"][-1]),
                  max_released_penetration_m=sim.max_depth, forbidden=bool(sim.ever_forbidden),
                  peak_force_n=float(trace["peak_force_n"].max()),
                  released_rope_force_n=float(np.max(trace["rope_force_n"][trace["released"]], initial=0)),
                  max_loaded_fingers=int(np.max(np.sum(trace["loads"] > .02, axis=1))))
    return report, trace
