"""S05 contact-style task; S02/S04 code and historical results stay unchanged."""

from dataclasses import fields

from flax import struct
import jax.numpy as jp
import numpy as np

from orca_suite_env import AuditedData, AuditedPhysics, OrcaSuiteGrasp
from s05_pad_contacts import pad_contacts


@struct.dataclass
class PadData(AuditedData):
    pad_audit: object


class PadPhysics(AuditedPhysics):
    def __init__(self, *args, **kwargs):
        self.pad_audit = np.zeros(6, np.float32)
        super().__init__(*args, **kwargs)

    def pack(self, data):
        base = super().pack(data)
        return PadData(**{f.name: getattr(base, f.name) for f in fields(AuditedData)},
                       pad_audit=self.pad_audit.copy())

    def sample_contacts(self, data):
        super().sample_contacts(data)
        good, bad, potential = pad_contacts(self.model, data)
        self.pad_audit[:2] = np.minimum(self.pad_audit[:2], good > .02)
        self.pad_audit[2:4] = np.maximum(self.pad_audit[2:4], bad > .02)
        self.pad_audit[4:] = potential

    def _single(self, mode, *arrays):
        self.pad_audit[:] = 0
        if mode == "forced_step":
            self.pad_audit[:2] = 1
        return super()._single(mode, *arrays)


class OrcaS05Grasp(OrcaSuiteGrasp):
    def __init__(self, model, arm, reference, *, pad_shaping=2., height_weight=0., **kwargs):
        super().__init__(model, arm, reference, audit=True, **kwargs)
        self.pad_shaping = float(pad_shaping)
        self.height_weight = float(height_weight)
        self.native = PadPhysics(self.mj_model, np.asarray(self.sensor_membership),
                                 self.contact_cone, self.n_substeps, self._forbidden_gids, True)

    def reset(self, rng):
        state = super().reset(rng)
        metrics = dict(state.metrics)
        for name in ("pad_pair", "pad_hold_sec", "best_pad_hold_sec", "pad_success30",
                     "pad_nonpad", "pad_potential", "pad_height_retention"):
            metrics[name] = jp.float32(0)
        return state.replace(metrics=metrics, info=dict(state.info, pad_steps=jp.int32(0),
                                                       best_pad_steps=jp.int32(0)))

    def step(self, state, action):
        nxt = super().step(state, action)
        audit, metrics = nxt.data.pad_audit, nxt.metrics
        pair = jp.all(audit[:2] > 0) & ~jp.any(audit[2:4] > 0)
        clean = (metrics["substep_forbidden"] == 0) & (metrics["substep_penetration"] <= .003)
        stable = pair & clean & (metrics["stable"] > 0)
        steps = jp.where(stable, state.info["pad_steps"] + 1, 0)
        best = jp.maximum(state.info["best_pad_steps"], steps)
        potential = jp.mean(audit[4:])
        retention = jp.where(pair & clean & (metrics["released"] > 0) & (nxt.done == 0),
                             jp.exp(-jp.maximum(nxt.info["cube_spawn_pos"][2] - metrics["cube_z"], 0) / .005), 0.)
        # Replace broad stable reward, retain the same drop cost. Dense feedback
        # is shared by pure/guided/residual policies and contains no demo angles.
        reward = (15. * stable - 50. * nxt.done + self.pad_shaping * potential
                  + self.height_weight * retention) * self.dt
        metrics = dict(metrics, pad_pair=pair.astype(jp.float32), pad_hold_sec=steps * self.dt,
                       best_pad_hold_sec=best * self.dt, pad_success30=(steps * self.dt >= 30).astype(jp.float32),
                       pad_nonpad=jp.max(audit[2:4]), pad_potential=potential, pad_height_retention=retention)
        return nxt.replace(reward=reward, metrics=metrics,
                           info=dict(nxt.info, pad_steps=steps, best_pad_steps=best))
