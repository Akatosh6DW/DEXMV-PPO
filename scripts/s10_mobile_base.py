"""S10: physical free-root hand following a Cartesian mocap target.

The weld drives the hand, never the cylinder. This is an ideal Cartesian
positioner, not a modeled robot arm or a torque-limited hardware controller.
All finger contacts and the S09 released-grasp gates remain native MuJoCo.
"""

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from s09_suspension import SuspendedCylinder, Suspension

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/orca_benchmark_suite/S09/inputs"
OUT = ROOT / "artifacts/orca_benchmark_suite/S10"
CENTER = np.array([.08, .055, .50])
LIFT = CENTER - np.array([.08, .055, .25])


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])


def quaternion(rot):
    return rot.as_quat()[[3, 0, 1, 2]]


def mobile_xml():
    root = ET.fromstring((INPUT / "model.xml").read_text())
    root.set("model", "S10 six-DoF Cartesian hand positioner")
    mount = root.find(".//body[@name='right_mount']")
    pos = np.fromstring(mount.get("pos"), sep=" ") + LIFT
    mount.set("pos", " ".join(map(str, pos)))
    # The original mount is massless because it was fixed to the world.
    ET.SubElement(mount, "freejoint", name="s10_hand_free")
    ET.SubElement(mount, "inertial", pos="0 0 0", mass=".1", diaginertia=".001 .001 .001")
    world = root.find("worldbody")
    ET.SubElement(world, "body", name="s10_target", mocap="true",
                  pos=mount.get("pos"), quat=mount.get("quat"))
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "weld", name="s10_positioner", body1="s10_target",
                  body2="right_mount", solref=".02 1", solimp=".95 .99 .001")
    return ET.tostring(root, encoding="unicode")


def align_rotation(source_quat, target_quat, rule="minimal", roll=0.):
    """Exploit axial/end-for-end cylinder symmetry, without future outcomes."""
    source_axis = rotation(source_quat).apply([0., -1., 0.])
    target_axis = rotation(target_quat).apply([0., -1., 0.])
    if np.dot(source_axis, target_axis) < 0:
        target_axis = -target_axis
    cross = np.cross(source_axis, target_axis)
    angle = np.arctan2(np.linalg.norm(cross), np.dot(source_axis, target_axis))
    align = (Rotation.identity() if np.linalg.norm(cross) < 1e-10 else
             Rotation.from_rotvec(cross / np.linalg.norm(cross) * angle))
    if rule == "gravity":
        # Keep the source frame's world-up direction as upright as axis alignment permits.
        up = np.array([0., 0., 1.])
        a = align.apply(up)
        a -= target_axis * np.dot(a, target_axis)
        b = up - target_axis * np.dot(up, target_axis)
        if np.linalg.norm(a) > 1e-8 and np.linalg.norm(b) > 1e-8:
            roll += np.arctan2(np.dot(target_axis, np.cross(a, b)), np.dot(a, b))
    elif rule != "minimal":
        raise ValueError(rule)
    return Rotation.from_rotvec(target_axis * roll) * align


class MobileCylinder(SuspendedCylinder):
    def __init__(self, xml=None, *, template=0, rule="minimal", roll=0., mobile=True,
                 prealigned=False, move_sec=2., release_sec=5.):
        super().__init__(mobile_xml() if xml is None else xml,
                         replace(Suspension(), release_sec=release_sec, episode_sec=42.))
        self.mount = self.model.body("right_mount").id
        joint = self.model.joint("s10_hand_free")
        self.root_qid, self.root_dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.target_id = int(self.model.body_mocapid[self.model.body("s10_target").id])
        self.original_pos = self.model.body_pos[self.mount].copy()
        self.original_rot = rotation(self.model.body_quat[self.mount])
        self.template, self.rule, self.roll = template, rule, roll
        self.mobile, self.prealigned, self.move_sec = mobile, prealigned, move_sec
        with np.load(INPUT / "reference.npz") as z:
            self.bank = dict(z)
        self.transform = Rotation.identity()

    def reset(self, position, quat, initial_ctrl=None, velocity=None):
        super().reset(position, quat, initial_ctrl, velocity)
        self.configure_plan(position, quat)
        self.data.qpos[self.root_qid:self.root_qid+7] = np.r_[self.home_pos, quaternion(self.home_rot)]
        self.data.mocap_pos[self.target_id] = self.home_pos
        self.data.mocap_quat[self.target_id] = quaternion(self.home_rot)
        mujoco.mj_forward(self.model, self.data)
        from probe_hic53_orca import cube_penetration
        self.initial_penetration = cube_penetration(self.model, self.data)
        return self.initial_penetration

    def configure_plan(self, position, quat):
        """Restore episode-specific planning state before every callback step."""
        self.transform = (align_rotation(self.bank["object_quaternion"][self.template], quat,
                                        self.rule, self.roll) if self.mobile else Rotation.identity())
        source_object = self.bank["object_position"][self.template] + LIFT
        self.goal_pos = np.asarray(position) + self.transform.apply(self.original_pos-source_object)
        self.goal_rot = self.transform * self.original_rot
        # A shared collision-free home pose, not an orientation-specific reset.
        self.home_pos = self.original_pos + np.array([0., -.20, 0.])
        self.home_rot = self.original_rot
        if not self.mobile:
            self.goal_pos, self.goal_rot = self.original_pos.copy(), self.original_rot
        if self.prealigned:
            self.home_pos, self.home_rot = self.goal_pos.copy(), self.goal_rot

    def target(self, time):
        u = np.clip(time / self.move_sec, 0, 1)
        smooth = u*u*(3-2*u)
        relative = self.goal_rot * self.home_rot.inv()
        rot = Rotation.from_rotvec(relative.as_rotvec()*smooth)*self.home_rot
        pos = self.home_pos + smooth*(self.goal_pos-self.home_pos)
        return pos, rot

    def reference(self, time):
        phase = max(0., time-self.move_sec)
        return np.array([np.interp(phase, self.bank["time"],
                                 self.bank["ctrl"][self.template, :, j]) for j in range(17)])

    def step(self, desired):
        pos, rot = self.target(self.data.time+self.cfg.control_dt)
        self.data.mocap_pos[self.target_id] = pos
        self.data.mocap_quat[self.target_id] = quaternion(rot)
        result = super().step(desired)
        actual = rotation(self.data.qpos[self.root_qid+3:self.root_qid+7])
        result.update(base_position_error_m=float(np.linalg.norm(self.data.qpos[self.root_qid:self.root_qid+3]-pos)),
                      base_rotation_error_rad=float((actual.inv()*rot).magnitude()))
        return result


def run_episode(sim, position, quat, velocity=None, seconds=42., keep_trace=False):
    sim.reset(position, quat, velocity=velocity)
    rows = []
    for _ in range(round(seconds/sim.cfg.control_dt)):
        row = sim.step(sim.reference(sim.data.time+sim.cfg.control_dt))
        if keep_trace:
            row.update(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy(), ctrl=sim.data.ctrl.copy(),
                       mocap=sim.data.mocap_pos.copy(), mocap_quat=sim.data.mocap_quat.copy())
        rows.append(row)
        if not row["alive"]:
            break
    report = dict(success30=bool(rows[-1]["success30"]), best_hold_sec=float(sim.best*sim.cfg.control_dt),
                  duration_sec=float(sim.data.time), initial_penetration_m=float(sim.initial_penetration),
                  max_released_penetration_m=float(sim.max_depth), forbidden=bool(sim.ever_forbidden),
                  peak_force_n=max(r["peak_force_n"] for r in rows),
                  max_path_penetration_m=max(r["penetration_m"] for r in rows),
                  final_base_position_error_m=rows[-1]["base_position_error_m"],
                  final_base_rotation_error_rad=rows[-1]["base_rotation_error_rad"],
                  released_rope_force_n=max((r["rope_force_n"] for r in rows if r["released"]), default=0.))
    return report, {k:np.array([r[k] for r in rows]) for k in rows[0]} if keep_trace else None
