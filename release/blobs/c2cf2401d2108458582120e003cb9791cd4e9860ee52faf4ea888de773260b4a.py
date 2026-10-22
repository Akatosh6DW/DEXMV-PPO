"""Explicitly terminate and record invalid numerical episodes, without retuning."""

from dataclasses import fields
import hashlib
from pathlib import Path

from flax import struct
import jax
import jax.numpy as jp
import numpy as np

from cylinder_s02_terminal_tail import TerminalData, TerminalPhysics, TerminalCylinderGrasp


@struct.dataclass
class FailureData(TerminalData):
    physics_invalid: object


class FailurePhysics(TerminalPhysics):
    def __init__(self, *args, evidence_dir, **kwargs):
        self.evidence_dir = Path(evidence_dir)
        self.events = []
        self.batch_index = ()
        super().__init__(*args, **kwargs)

    def pack(self, data):
        base = super().pack(data)
        return FailureData(**{f.name: getattr(base, f.name) for f in fields(TerminalData)},
                           physics_invalid=np.asarray(False))

    def _single(self, mode, *arrays):
        if mode != 'terminal_guard_step' or not bool(arrays[-1][0]):
            return super()._single(mode, *arrays)
        try:
            return super()._single(mode, *arrays)
        except FloatingPointError as error:
            warnings = np.array(self.scratch.warning.number, copy=True)
            words, ctrl, mocap, _force, _active = arrays
            data = self.restore(words)
            qpos, qvel = data.qpos.copy(), data.qvel.copy()
            event = dict(batch_index=list(self.batch_index), incoming_time=float(data.time),
                         error=str(error), warning_counts=warnings.tolist(),
                         input_sha256=hashlib.sha256(np.asarray(words).tobytes()).hexdigest())
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            filename = 'event_%04d.npz' % len(self.events)
            np.savez_compressed(self.evidence_dir/filename, qpos=qpos, qvel=qvel,
                                **{'argument_%d' % i: a for i, a in enumerate(arrays)})
            event['state_file'] = filename
            self.events.append(event)
            # A finite pre-error pose is a display placeholder, not a valid
            # simulated step. The evaluator excludes this step from metrics.
            result = super()._single('forward', words, qpos, qvel, ctrl, mocap)
            return result.replace(physics_invalid=np.asarray(True), episode_active=np.asarray(False))

    def batched(self, mode, *arrays):
        arrays = tuple(np.asarray(x) for x in arrays)
        shape = arrays[0].shape[:-1]
        with self.lock:
            if not shape:
                self.batch_index = ()
                return self.single(mode, *arrays)
            values = []
            for index in np.ndindex(shape):
                self.batch_index = index
                values.append(self.single(mode, *(a[index] for a in arrays)))
            return jax.tree_util.tree_map(
                lambda *items: np.stack(items).reshape((*shape, *items[0].shape)), *values)


class FailureCylinderGrasp(TerminalCylinderGrasp):
    evidence_dir = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.evidence_dir is None:
            raise ValueError('Numerical failures must have a persistent evidence directory')
        self.native = FailurePhysics(self.mj_model, np.asarray(self.sensor_membership),
                                     self.contact_cone, self.n_substeps, self._forbidden_gids,
                                     True, evidence_dir=self.evidence_dir)

    def step(self, state, action):
        nxt = super().step(state, action)
        invalid = state.data.physics_invalid | nxt.data.physics_invalid
        return nxt.replace(done=jp.where(invalid, jp.float32(1), nxt.done),
                           data=nxt.data.replace(physics_invalid=invalid,
                                                 episode_active=nxt.data.episode_active & ~invalid))
