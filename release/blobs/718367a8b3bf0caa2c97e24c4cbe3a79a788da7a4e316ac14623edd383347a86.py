"""P03 longitudinal palm-wrap reference exposed to the matched PPO interface.

This reference is geometry-derived, not a newly extracted video demonstration.
The P03 contact criterion remains separate from S07's distal-only task.
"""

from dataclasses import fields

from flax import struct
import jax.numpy as jp
import mujoco
import numpy as np

from orca_s05_env import PadData, PadPhysics
from orca_s07_env import OrcaS07Grasp
from orca_suite_env import AuditedPhysics
from probe_hic53_orca import cube_penetration
from probe_s08_wrap import contacts


@struct.dataclass
class WrapData(PadData):
    wrap_loads: object


class WrapPhysics(PadPhysics):
    def __init__(self, *args, **kwargs):
        self.loads = np.zeros(6, np.float32)
        super().__init__(*args, **kwargs)

    def pack(self, data):
        # Also measure reset penetration, before any control step can hide it.
        self.audit[0] = max(self.audit[0], cube_penetration(self.model, data))
        base = super().pack(data)
        return WrapData(**{f.name: getattr(base, f.name) for f in fields(PadData)},
                        wrap_loads=self.loads.copy())

    def _single(self, mode, *arrays):
        self.loads[:] = np.inf if mode == "forced_step" else 0
        return super()._single(mode, *arrays)

    def sample_contacts(self, data):
        AuditedPhysics.sample_contacts(self, data)
        good, bad, peak = contacts(self.model, data)
        self.loads = np.minimum(self.loads, good).astype(np.float32)
        self.pad_audit[:2] = np.minimum(self.pad_audit[:2],
            [good[0] > .02 and good[1] > .02, np.sum(good[2:] > .02) >= 2])
        self.audit[1] = max(self.audit[1], float(bad))
        # Retain S07's group-summed peak convention for cross-method comparisons.
        self.audit[2] = max(self.audit[2], peak)
        # Dense geometry-only shaping: contact coverage, no reference joint angles.
        self.pad_audit[4:] = [np.mean(np.minimum(good[:2]/.02, 1)),
                             min(np.sum(np.minimum(good[2:]/.02, 1))/2, 1)]


class OrcaS08Grasp(OrcaS07Grasp):
    criterion = "S08 P03 wrap: palm+thumb+two fingers, curved palmar load, initial<=1mm, height loss<15mm,30s"

    def __init__(self, model, arm, reference, **kwargs):
        super().__init__(model, arm, reference, **kwargs)
        # Palm and all hand surfaces may assist. Only non-hand contacts are forbidden.
        self._forbidden_gids = [i for i in range(self.mj_model.ngeom)
            if self.mj_model.geom_contype[i] != 1 and i != self._cube_geom_id]
        self.native = WrapPhysics(self.mj_model, np.asarray(self.sensor_membership),
            self.contact_cone, self.n_substeps, self._forbidden_gids, True)
        with self._config.unlocked():
            self._config.s08 = dict(criterion=self.criterion, reference_origin="manual geometry P03",
                pad_shaping=self.pad_shaping, height_weight=self.height_weight,
                reward="dt*(15*valid_wrap -50*done +pad_shaping*coverage +height_weight*height_retention)")

    def reset(self, rng):
        state = super().reset(rng)
        depth = state.data.substep_audit[0]
        return state.replace(info=dict(state.info, initial_penetration=depth),
            metrics=dict(state.metrics, initial_penetration=depth, wrap_palm=jp.float32(0),
                         wrap_thumb=jp.float32(0), wrap_other_count=jp.float32(0)))

    def step(self, state, action):
        nxt = super().step(state, action)
        m, loads = nxt.metrics, nxt.data.wrap_loads
        pair = (loads[0]>.02)&(loads[1]>.02)&(jp.sum(loads[2:]>.02)>=2)
        clean = (m["substep_forbidden"]==0)&(m["substep_penetration"]<=.003)
        stable = pair & clean & (m["released"]>0) & (nxt.done==0)
        stable &= (m["cube_z"] > nxt.info["cube_spawn_pos"][2]-.015)
        stable &= state.info["initial_penetration"] <= .001
        steps = jp.where(stable, state.info["pad_steps"]+1, 0)
        best = jp.maximum(state.info["best_pad_steps"], steps)
        coverage = jp.mean(nxt.data.pad_audit[4:])
        retention = jp.where(pair & clean & (m["released"]>0) & (nxt.done==0),
            jp.exp(-jp.maximum(nxt.info["cube_spawn_pos"][2]-m["cube_z"],0)/.005),0.)
        reward = (15*stable-50*nxt.done+self.pad_shaping*coverage+self.height_weight*retention)*self.dt
        metrics = dict(m, stable=stable.astype(jp.float32), hold_sec=steps*self.dt,
            best_hold_sec=best*self.dt, success30=(best*self.dt>=30).astype(jp.float32),
            pad_pair=pair.astype(jp.float32), pad_hold_sec=steps*self.dt,
            best_pad_hold_sec=best*self.dt, pad_success30=(steps*self.dt>=30).astype(jp.float32),
            pad_nonpad=jp.float32(0), pad_potential=coverage, pad_height_retention=retention,
            initial_penetration=state.info["initial_penetration"], wrap_palm=loads[0],
            wrap_thumb=loads[1], wrap_other_count=jp.sum(loads[2:]>.02).astype(jp.float32))
        info = dict(nxt.info,pad_steps=steps,best_pad_steps=best,
                    stable_hold_steps=steps,best_hold_steps=best)
        return nxt.replace(reward=reward,metrics=metrics,info=info,obs=self._obs(nxt.data,info))
