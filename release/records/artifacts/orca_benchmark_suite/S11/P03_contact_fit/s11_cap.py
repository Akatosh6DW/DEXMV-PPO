"""S11 feasibility: passive cap, fixed bottle, physically movable hand root."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from oakink_cap_dynamics import contact_report
from orca_contact_forces import normal_loads

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/orca_formal/CAP21_thread_stop_and_filter"
REFERENCE = ROOT / "artifacts/orca_formal/CAP10_sequence_mount_ik/kinematic_reference.npz"
OUT = ROOT / "artifacts/orca_benchmark_suite/S11"


def mobile_xml():
    root = ET.fromstring((SOURCE / "model.xml").read_text())
    root.set("model", "S11 passive cap with Cartesian hand positioner")
    mount = root.find(".//body[@name='right_mount']")
    ET.SubElement(mount, "freejoint", name="s11_hand_free")
    ET.SubElement(mount, "inertial", pos="0 0 0", mass=".1", diaginertia=".001 .001 .001")
    ET.SubElement(root.find("worldbody"), "body", name="s11_target", mocap="true",
                  pos=mount.get("pos"), quat=mount.get("quat"))
    ET.SubElement(root.find("equality"), "weld", name="s11_positioner", body1="s11_target",
                  body2="right_mount", solref=".02 1", solimp=".95 .99 .001")
    return ET.tostring(root, encoding="unicode")


def smooth(value):
    value = np.clip(value, 0., 1.)
    return value*value*(3.-2.*value)


class CapPilot:
    def __init__(self, *, moving=True, grip_time=0., twist_seconds=14.):
        self.xml = mobile_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.meta = json.loads((SOURCE / "report.json").read_text())
        self.moving, self.grip_time, self.twist_seconds = moving, grip_time, twist_seconds
        self.dt, self.duration = .05, 24.
        self.substeps = round(self.dt/self.model.opt.timestep)
        self.thread_ids = np.array([i for i in range(self.model.neq)
                                    if self.model.equality(i).name.startswith("thread_")])
        self.hand_eq = self.model.equality("s11_positioner").id
        self.target_id = int(self.model.body_mocapid[self.model.body("s11_target").id])
        self.root_qid = int(self.model.joint("s11_hand_free").qposadr[0])
        self.turn = int(self.model.joint("cap_turn").qposadr[0])
        self.cap = self.model.body("oakink_cap").id
        self.bottle = self.model.body("oakink_bottle").id
        self.qids = self.model.jnt_qposadr[self.model.actuator_trnid[:,0]]
        self.cap_mask = (self.model.geom_bodyid==self.cap) & (self.model.geom_contype==2)
        self.bottle_mask = (self.model.geom_bodyid==self.bottle) & (self.model.geom_contype==4)
        self.fingers = np.array([[(self.model.geom(g).name or "").startswith(f"orca_{f}_")
                                 for g in range(self.model.ngeom)]
                                for f in ("thumb","index","middle","ring","pinky")])
        self.release_angle = self.meta["parameters"]["release_angle_rad"]
        self.pitch = self.meta["parameters"]["pitch_m_per_rad"]
        with np.load(REFERENCE) as packet:
            anchor = np.argmin(abs(packet["source_frame_id"]-self.meta["source_anchor_frame"]))
            self.ref_time = packet["time"][anchor:]-packet["time"][anchor]
            self.ref_ctrl = packet["ctrl"][anchor:].copy()
        assert len(self.thread_ids)==5 and self.model.nu==17

    def thread_mode(self):
        # The hand weld remains active when the five passive thread constraints release.
        if self.data.qpos[self.turn]>=self.release_angle:
            self.data.eq_active[self.thread_ids]=False
        engaged = bool(self.data.eq_active[self.thread_ids].any())
        self.model.jnt_limited[self.model.joint("cap_turn").id] = engaged
        self.model.geom_conaffinity[self.cap_mask] = 1 if engaged else 5
        self.model.geom_conaffinity[self.bottle_mask] = 1 if engaged else 3
        return engaged

    def reset(self, seed):
        mujoco.mj_resetData(self.model,self.data)
        rng=np.random.default_rng(seed)
        initial=np.asarray(self.meta["initial_ctrl"])
        self.data.qpos[self.qids]=np.clip(initial+rng.normal(0,.002,17),*self.model.actuator_ctrlrange.T)
        self.data.ctrl[:]=initial
        self.thread_mode()
        mujoco.mj_forward(self.model,self.data)
        self.home_pos=self.data.qpos[self.root_qid:self.root_qid+3].copy()
        q=self.data.qpos[self.root_qid+3:self.root_qid+7]
        self.home_rot=Rotation.from_quat(q[[1,2,3,0]])
        self.center=self.data.xipos[self.cap].copy()
        self.axis=self.data.xmat[self.bottle].reshape(3,3)[:,2].copy()
        self.data.mocap_pos[self.target_id]=self.home_pos
        self.data.mocap_quat[self.target_id]=q
        depths=contact_report(self.model,self.data)
        self.initial_depth=max(depths["hand_cap_depth_m"],depths["hand_bottle_depth_m"])
        self.max_depth=self.initial_depth
        self.released_at=None
        self.frozen_angle=0.
        self.hold=self.best_hold=0
        self.ever_bad=False
        self.max_force=0.

    def theta(self,time):
        return (self.release_angle+.6)*smooth((time-2.)/self.twist_seconds)

    def target(self,time):
        if not self.moving:
            return self.home_pos,self.home_rot
        angle=self.theta(time) if self.released_at is None else self.frozen_angle
        lift=0. if self.released_at is None else .05*smooth((time-self.released_at)/3.)
        rotation=Rotation.from_rotvec(self.axis*angle)
        pos=self.center+rotation.apply(self.home_pos-self.center)+self.axis*(self.pitch*angle+lift)
        return pos,rotation*self.home_rot

    def contacts(self):
        loads=np.zeros(5)
        bad=bottle=floor=False
        peak=0.
        contacts=self.data.contact[:self.data.ncon]
        forces=normal_loads(contacts.efc_address,self.data.efc_force,"elliptic")
        for c,force in zip(contacts,forces):
            a,b=map(int,c.geom)
            if c.dist>0 or force<=1e-5 or self.cap_mask[a]==self.cap_mask[b]:
                continue
            other=b if self.cap_mask[a] else a
            is_finger=self.fingers[:,other]
            loads+=is_finger*force
            bottle |= self.model.geom_bodyid[other]==self.bottle
            floor |= self.model.geom(other).name=="floor"
            bad |= self.model.geom_contype[other]==1 and not is_finger.any()
            if is_finger.any():
                peak=max(peak,float(force))
        return loads,bad,bottle,floor,peak

    def step(self):
        time=self.data.time+self.dt
        pos,rot=self.target(time)
        self.data.mocap_pos[self.target_id]=pos
        self.data.mocap_quat[self.target_id]=rot.as_quat()[[3,0,1,2]]
        phase=min(time,self.grip_time)
        desired=np.array([np.interp(phase,self.ref_time,self.ref_ctrl[:,j]) for j in range(17)])
        desired=np.clip(desired,*self.model.actuator_ctrlrange.T)
        self.data.ctrl[:]=np.clip(desired,self.data.ctrl-.1,self.data.ctrl+.1)
        loads=np.full(5,np.inf)
        bad=bottle=floor=False
        depth=peak=0.
        for _ in range(self.substeps):
            engaged=self.thread_mode()
            if not engaged and self.released_at is None:
                self.released_at=float(self.data.time)
                self.frozen_angle=self.theta(self.data.time)
            assert self.data.eq_active[self.hand_eq]
            assert not np.any(self.data.xfrc_applied) and not np.any(self.data.qfrc_applied)
            mujoco.mj_step(self.model,self.data)
            current,b,c,f,p=self.contacts()
            loads=np.minimum(loads,current)
            bad|=b; bottle|=c; floor|=f
            peak=max(peak,p)
            d=contact_report(self.model,self.data)
            depth=max(depth,d["hand_cap_depth_m"],d["hand_bottle_depth_m"])
        if not np.isfinite(self.data.qpos).all() or np.any(self.data.warning.number):
            raise FloatingPointError("S11 physics warning")
        self.max_depth=max(self.max_depth,depth)
        self.max_force=max(self.max_force,peak)
        self.ever_bad|=bad
        lift=float((self.data.xipos[self.cap]-self.center)@self.axis)
        released=not bool(self.data.eq_active[self.thread_ids].any())
        paired=bool(loads[0]>.02 and np.any(loads[1:]>.02))
        valid=bool(released and lift>=.018 and paired and not (bad or bottle or floor)
                   and self.initial_depth<=.001 and self.max_depth<=.003 and not self.ever_bad)
        self.hold=self.hold+1 if valid else 0
        self.best_hold=max(self.best_hold,self.hold)
        return dict(time=float(self.data.time),angle=float(self.data.qpos[self.turn]),lift=lift,
                    released=released,paired=paired,valid=valid,hold_sec=self.hold*self.dt,
                    bad=bool(bad),bottle_contact=bool(bottle),floor_contact=bool(floor),
                    depth_m=depth,peak_force_n=peak,loads=loads.copy(),
                    success=bool(self.hold*self.dt>=1 and self.max_force<=30.),
                    qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),ctrl=self.data.ctrl.copy(),
                    mocap_pos=self.data.mocap_pos.copy(),mocap_quat=self.data.mocap_quat.copy())
