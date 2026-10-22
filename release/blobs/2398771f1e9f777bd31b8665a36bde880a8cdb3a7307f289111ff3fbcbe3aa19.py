"""Evaluation-only absorbing physics after an episode has already terminated.

The frozen evaluator masks terminated episodes, but still calls env.step on
their diagnostic tails. Do not integrate those tails into numerical overflow.
Every active step, including the first terminal step, uses original physics.
"""

from dataclasses import fields

from flax import struct
import numpy as np

from orca_cylinder_s02_env import OrcaCylinderS02Grasp
from orca_suite_env import AuditedData, AuditedPhysics


@struct.dataclass
class TerminalData(AuditedData):
    episode_active: object


class TerminalPhysics(AuditedPhysics):
    def pack(self, data):
        base = super().pack(data)
        return TerminalData(**{f.name: getattr(base, f.name) for f in fields(AuditedData)},
                            episode_active=np.asarray(True))

    def _single(self, mode, *arrays):
        if mode != 'terminal_guard_step':
            return super()._single(mode, *arrays)
        words, ctrl, mocap, force, active = arrays
        if bool(active[0]):
            return super()._single('forced_step', words, ctrl, mocap, force)
        # Preserve a finite terminal pose. Updating controls and kinematics
        # keeps existing trace interfaces; no post-terminal integration occurs.
        data = self.restore(words)
        return super()._single('forward', words, data.qpos.copy(), data.qvel.copy(), ctrl, mocap)


class TerminalCylinderGrasp(OrcaCylinderS02Grasp):
    def __init__(self, *args, **kwargs):
        if not kwargs.get('audit', False):
            raise ValueError('Terminal-tail recovery is formal-evaluation-only')
        super().__init__(*args, **kwargs)
        self.native = TerminalPhysics(self.mj_model, np.asarray(self.sensor_membership),
                                      self.contact_cone, self.n_substeps,
                                      self._forbidden_gids, True)

    def _step_physics(self, data, ctrl):
        return self.native.call('terminal_guard_step', data.integration_words, ctrl,
                                data.mocap_pos, data.external_force,
                                data.episode_active[..., None])

    def step(self, state, action):
        nxt = super().step(state, action)
        active = state.data.episode_active & (nxt.done == 0)
        return nxt.replace(data=nxt.data.replace(episode_active=active))
