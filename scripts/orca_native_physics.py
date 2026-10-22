"""Native MuJoCo stepping behind the unchanged matched-task/Brax interface.

All native integration state, including warm starts, is carried explicitly.
Double-precision state is serialized as uint32 words so JAX does not truncate it
when x64 is disabled. Scratch storage is reused only after a complete restore.
"""

import threading

from flax import struct
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from orca_contact_forces import normal_loads
from orca_matched_env import OrcaMatchedGrasp, finger_normal_forces

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


@struct.dataclass
class NativeData:
    integration_words: object
    qpos: object
    qvel: object
    ctrl: object
    mocap_pos: object
    xpos: object
    geom_xpos: object
    actuator_force: object
    cube_contacts: object
    finger_force: object


class NativePhysics:
    def __init__(self, model, membership, cone, substeps):
        self.model, self.membership, self.cone, self.substeps = model, membership, cone, substeps
        self.cube = model.geom("cube").id
        self.state_size = mujoco.mj_stateSize(model, STATE_SPEC)
        # XLA callback threads have ephemeral Python thread states. A thread-local
        # MjData destructor can run after its thread state is gone (MuJoCo 3.5).
        self.scratch = mujoco.MjData(model)
        self.lock = threading.RLock()
        example = self.pack(mujoco.MjData(model))
        self.output_spec = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), example)

    def scratch_data(self):
        return self.scratch

    def pack(self, data):
        state = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, data, state, STATE_SPEC)
        contact = data.contact[:data.ncon]
        force = normal_loads(contact.efc_address, data.efc_force, self.cone)
        active = (contact.dist <= 0) & (force > 1e-5)
        cube_contacts = np.zeros(self.model.ngeom, dtype=bool)
        for pair, enabled in zip(contact.geom, active):
            if enabled and self.cube in pair:
                other = pair[1] if pair[0] == self.cube else pair[0]
                cube_contacts[other] = True
        sensed = finger_normal_forces(contact.geom, contact.efc_address, data.efc_force,
                                       self.membership, cone=self.cone)
        return NativeData(state.view(np.uint32).copy(), data.qpos.astype(np.float32),
                          data.qvel.astype(np.float32), data.ctrl.astype(np.float32),
                          data.mocap_pos.astype(np.float32), data.xpos.astype(np.float32),
                          data.geom_xpos.astype(np.float32), data.actuator_force.astype(np.float32),
                          cube_contacts, sensed.astype(np.float32))

    def restore(self, words):
        state = np.ascontiguousarray(words, dtype=np.uint32).view(np.float64)
        if state.shape != (self.state_size,):
            raise ValueError("Native integration state size changed")
        data = self.scratch_data()
        mujoco.mj_setState(self.model, data, state, STATE_SPEC)
        return data

    def single(self, mode, *arrays):
        with self.lock:
            return self._single(mode, *arrays)

    def _single(self, mode, *arrays):
        if mode == "reset":
            data = self.scratch_data()
            mujoco.mj_resetData(self.model, data)
            data.qpos[:], data.qvel[:], data.ctrl[:], data.mocap_pos[:] = arrays
            mujoco.mj_forward(self.model, data)
        elif mode == "forward":
            words, qpos, qvel, ctrl, mocap = arrays
            data = self.restore(words)
            data.warning.number[:] = 0
            data.qpos[:], data.qvel[:], data.ctrl[:], data.mocap_pos[:] = qpos, qvel, ctrl, mocap
            mujoco.mj_forward(self.model, data)
        elif mode == "step":
            words, ctrl, mocap = arrays
            data = self.restore(words)
            data.warning.number[:] = 0
            data.ctrl[:], data.mocap_pos[:] = ctrl, mocap
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, data)
        else:
            raise ValueError(mode)
        bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
               mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
        if any(data.warning[int(kind)].number for kind in bad):
            raise FloatingPointError("MuJoCo warned about invalid physics or silently reset the simulation")
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("Nonfinite native physics")
        return self.pack(data)

    def batched(self, mode, *arrays):
        arrays = tuple(np.asarray(x) for x in arrays)
        batch_shape = arrays[0].shape[:-1]
        if not batch_shape:
            return self.single(mode, *arrays)
        values = [self.single(mode, *(a[index] for a in arrays)) for index in np.ndindex(batch_shape)]
        return jax.tree_util.tree_map(
            lambda *items: np.stack(items).reshape((*batch_shape, *items[0].shape)), *values)

    def call(self, mode, *arrays):
        return jax.pure_callback(lambda *a: self.batched(mode, *a), self.output_spec,
                                 *arrays, vmap_method="broadcast_all")


class OrcaNativeMatchedGrasp(OrcaMatchedGrasp):
    physics_backend = "native_mujoco_callback"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native = NativePhysics(self.mj_model, np.asarray(self.sensor_membership), self.contact_cone, self.n_substeps)

    def _make_initial_data(self, *, qpos, qvel, ctrl, mocap_pos):
        return self.native.call("reset", qpos, qvel, ctrl, mocap_pos)

    def _forward_physics(self, data):
        return self.native.call("forward", data.integration_words, data.qpos, data.qvel, data.ctrl, data.mocap_pos)

    def _step_physics(self, data, ctrl):
        return self.native.call("step", data.integration_words, ctrl, data.mocap_pos)

    def _has_cube_contact(self, data, geom_ids):
        return jp.any(data.cube_contacts[jp.asarray(geom_ids)]).astype(jp.float32)

    def _forces(self, data):
        return data.finger_force
