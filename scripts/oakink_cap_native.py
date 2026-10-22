"""Explicit-state native physics for the separate, source-matched cap task."""

import threading

from flax import struct
import jax
import mujoco
import numpy as np

from oakink_cap_dynamics import apply_thread_contact_mode, contact_report, step_passive
from orca_contact_forces import normal_loads
from orca_matched_env import finger_normal_forces
from orca_native_physics import NativePhysics, STATE_SPEC


@struct.dataclass
class CapData:
    integration_words: object
    qpos: object
    qvel: object
    ctrl: object
    xpos: object
    xquat: object
    geom_xpos: object
    cap_center: object
    actuator_force: object
    finger_force: object
    cap_fingers: object
    forbidden: object
    bottle_contact: object
    floor_contact: object
    hand_penetration: object
    thread_active: object


class CapNativePhysics(NativePhysics):
    def __init__(self, model, release_angle, substeps=25):
        self.model, self.release_angle, self.substeps = model, release_angle, substeps
        self.cap = model.body("oakink_cap").id
        self.bottle = model.body("oakink_bottle").id
        self.membership = np.array([[model.geom(g).name.startswith(f"orca_{finger}_")
                                     for g in range(model.ngeom)]
                                    for finger in ("thumb", "index", "middle", "ring", "pinky")])
        self.cap_geoms = set(np.flatnonzero((model.geom_bodyid == self.cap) & (model.geom_contype == 2)))
        self.state_size = mujoco.mj_stateSize(model, STATE_SPEC)
        self.scratch = mujoco.MjData(model)
        self.lock = threading.RLock()
        self.cone = "elliptic" if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else "pyramidal"
        example = self.pack(self.scratch)
        self.output_spec = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), example)

    def pack(self, data):
        state = np.empty(self.state_size, np.float64)
        mujoco.mj_getState(self.model, data, state, STATE_SPEC)
        contacts = data.contact[:data.ncon]
        force = normal_loads(contacts.efc_address, data.efc_force, self.cone)
        fingers = np.zeros(5, bool)
        forbidden = bottle = floor = False
        for c, normal in zip(contacts, force):
            if c.dist > 0 or normal <= 1e-5:
                continue
            a, b = int(c.geom1), int(c.geom2)
            if a in self.cap_geoms and b in self.cap_geoms:
                continue
            if a not in self.cap_geoms and b not in self.cap_geoms:
                continue
            other = b if a in self.cap_geoms else a
            fingers |= self.membership[:, other]
            bottle |= self.model.geom_bodyid[other] == self.bottle
            floor |= self.model.geom(other).name == "floor"
            forbidden |= self.model.geom_contype[other] == 1 and not self.membership[:, other].any()
        depths = contact_report(self.model, data)
        touch = finger_normal_forces(contacts.geom, contacts.efc_address, data.efc_force,
                                     self.membership, cone=self.cone)
        return CapData(state.view(np.uint32).copy(), data.qpos.astype(np.float32), data.qvel.astype(np.float32),
                       data.ctrl.astype(np.float32), data.xpos.astype(np.float32), data.xquat.astype(np.float32),
                       data.geom_xpos.astype(np.float32), data.xipos[self.cap].astype(np.float32),
                       data.actuator_force.astype(np.float32), touch.astype(np.float32), fingers,
                       np.asarray(forbidden), np.asarray(bottle), np.asarray(floor),
                       np.float32(max(depths["hand_cap_depth_m"], depths["hand_bottle_depth_m"])),
                       np.asarray(data.eq_active.any()))

    def _single(self, mode, *arrays):
        if mode == "reset":
            data = self.scratch_data()
            mujoco.mj_resetData(self.model, data)
            data.qpos[:], data.ctrl[:] = arrays
            apply_thread_contact_mode(self.model, data)
            mujoco.mj_forward(self.model, data)
        elif mode == "step":
            words, ctrl = arrays
            data = self.restore(words)
            data.warning.number[:] = 0
            if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                raise ValueError("Policy task forbids external cap forces/torques")
            data.ctrl[:] = ctrl
            step_passive(self.model, data, self.release_angle, self.substeps)
        else:
            raise ValueError(mode)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("Nonfinite cap physics")
        return self.pack(data)
