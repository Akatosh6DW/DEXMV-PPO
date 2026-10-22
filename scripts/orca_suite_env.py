"""Versioned benchmark variations; never mutate the frozen M01 environment."""

from dataclasses import fields

from flax import struct
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from orca_acquisition_env import OrcaAcquisitionGrasp
from orca_native_physics import NativeData, NativePhysics
from orca_residual_controls import load_reference
from orca_matched_env import finger_normal_forces
from orca_contact_forces import normal_loads


BENCHMARKS = {
    "B0_nominal": dict(jitter=.002),
    "B1_position": dict(jitter=.006),
    "B2_yaw": dict(jitter=.002, yaw_degrees=10.),
    "B3_friction": dict(jitter=.002, friction_scale=.6),
    "B4_mass": dict(jitter=.002, mass_scale=2.),
    "B5_push": dict(jitter=.002, push_force=.15),
}


@struct.dataclass
class AuditedData(NativeData):
    substep_audit: object
    external_force: object


class AuditedPhysics(NativePhysics):
    def __init__(self, model, membership, cone, substeps, forbidden, enabled):
        self.audit_enabled = enabled
        self.audit = np.zeros(3, np.float32)
        self.forbidden = np.asarray(forbidden)
        self.cube_body = model.body("cube").id
        super().__init__(model, membership, cone, substeps)

    def pack(self, data):
        base = super().pack(data)
        return AuditedData(**{f.name: getattr(base, f.name) for f in fields(NativeData)},
                           substep_audit=self.audit.copy(),
                           external_force=data.xfrc_applied[self.cube_body, :3].astype(np.float32))

    def sample_contacts(self, data):
        contact = data.contact[:data.ncon]
        pairs = contact.geom
        other = np.where(pairs[:, 0] == self.cube, pairs[:, 1], pairs[:, 0])
        touches = np.any(pairs == self.cube, axis=1)
        hand = touches & (self.model.geom_contype[other] == 1)
        depth = float(np.max(np.where(hand, -contact.dist, 0.), initial=0.))
        force = normal_loads(contact.efc_address, data.efc_force, self.cone)
        forbidden = np.any(touches & np.isin(other, self.forbidden) &
                           (contact.dist <= 0) & (force > 1e-5))
        peak = np.max(finger_normal_forces(pairs, contact.efc_address, data.efc_force,
                                          self.membership, cone=self.cone), initial=0.)
        self.audit = np.maximum(self.audit, [depth, forbidden, peak]).astype(np.float32)

    def _single(self, mode, *arrays):
        self.audit[:] = 0
        if mode != "forced_step":
            return super()._single(mode, *arrays)
        words, ctrl, mocap, force = arrays
        data = self.restore(words)
        data.warning.number[:] = 0
        data.ctrl[:], data.mocap_pos[:] = ctrl, mocap
        data.xfrc_applied[:] = 0
        data.xfrc_applied[self.cube_body, :3] = force
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, data)
            if self.audit_enabled:
                self.sample_contacts(data)
        bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
               mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
        if any(data.warning[int(k)].number for k in bad):
            raise FloatingPointError("Invalid physics or automatic MuJoCo reset")
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("Nonfinite physics")
        return self.pack(data)


class OrcaSuiteGrasp(OrcaAcquisitionGrasp):
    def __init__(self, model, arm, reference, *, benchmark="B0_nominal",
                 variant="video", raw_reference=None, audit=False, residual_gain=1.):
        if benchmark not in BENCHMARKS or variant not in ("video", "linear", "unrefined"):
            raise ValueError("Unknown benchmark/reference")
        if not 0 <= residual_gain <= 1:
            raise ValueError("Invalid residual intervention")
        self.benchmark = benchmark
        self.settings = dict(BENCHMARKS[benchmark])
        self.residual_gain = residual_gain
        super().__init__(model, arm, reference, jitter=self.settings["jitter"])
        if arm == "residual":
            if variant == "linear":
                fraction = np.minimum((np.arange(800) + 1) * self.dt / 1.5, 1)[:, None]
                initial = np.asarray(self._config.initial_ctrl)
                self.reference_ctrl = jp.asarray(initial + fraction * (np.asarray(self.reference_ctrl[-1]) - initial))
            elif variant == "unrefined":
                if raw_reference is None:
                    raise ValueError("Raw reference required")
                reference_ctrl, _ = load_reference(raw_reference, self._lowers, self._uppers,
                                                   self._config.initial_ctrl, self.dt, 800, .1)
                self.reference_ctrl = jp.asarray(reference_ctrl)
        elif variant != "video" or residual_gain != 1:
            raise ValueError("Reference interventions require residual arm")
        physical = self.mj_model.geom_contype != 0
        self.mj_model.geom_friction[physical, 0] *= self.settings.get("friction_scale", 1.)
        cube_id = self.mj_model.body("cube").id
        self.mj_model.body_mass[cube_id] *= self.settings.get("mass_scale", 1.)
        self.mj_model.body_inertia[cube_id] *= self.settings.get("mass_scale", 1.)
        mujoco.mj_setConst(self.mj_model, mujoco.MjData(self.mj_model))
        self.native = AuditedPhysics(self.mj_model, np.asarray(self.sensor_membership),
                                     self.contact_cone, self.n_substeps, self._forbidden_gids, audit)

    def reset(self, rng):
        state = super().reset(rng)
        yaw = jax.random.uniform(jax.random.fold_in(rng, 661), (), minval=-1., maxval=1.)
        yaw *= self.settings.get("yaw_degrees", 0.) * np.pi / 180
        w, x, y, z = self.object_quaternion
        c, s = jp.cos(yaw / 2), jp.sin(yaw / 2)
        quat = jp.array([c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w])
        data = state.data.replace(qpos=state.data.qpos.at[self._cube_qids[3:]].set(quat))
        # World-Z rotation preserves support height. Preserve all other reset RNGs.
        data = self._forward_physics(data)
        angle = jax.random.uniform(jax.random.fold_in(rng, 662), (), minval=0., maxval=2*np.pi)
        direction = jp.array([jp.cos(angle), jp.sin(angle), 0.])
        info = dict(state.info, push_direction=direction, applied_push=jp.zeros(3))
        metrics = dict(state.metrics, substep_penetration=jp.float32(0),
                       substep_forbidden=jp.float32(0), substep_force_peak=jp.float32(0))
        return state.replace(data=data, info=info, metrics=metrics, obs=self._obs(data, info))

    def _step_physics(self, data, ctrl):
        # Force is passed explicitly, never kept as mutable episode-global state.
        return self.native.call("forced_step", data.integration_words, ctrl, data.mocap_pos,
                                data.external_force)

    def step(self, state, action):
        t = state.info["step"] * self.dt
        push = self.settings.get("push_force", 0.) * state.info["push_direction"]
        force = jp.where((t >= 10.) & (t < 10.2), push, jp.zeros(3))
        incoming = state.replace(data=state.data.replace(external_force=force))
        nxt = super().step(incoming, action * self.residual_gain if self.arm == "residual" else action)
        metrics = dict(nxt.metrics, substep_penetration=nxt.data.substep_audit[0],
                       substep_forbidden=nxt.data.substep_audit[1],
                       substep_force_peak=nxt.data.substep_audit[2])
        return nxt.replace(metrics=metrics, info=dict(nxt.info, applied_push=force))
