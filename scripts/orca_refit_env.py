"""Versioned, instrumented rewards on an explicitly selected collision model."""

import math

import jax.numpy as jp

from orca_residual_env import OrcaResidualGrasp

BASE_WEIGHTS = dict(stable=15., done=-50., speed=0., pair=0., hold_progress=0., correction=0.)


class OrcaRefitGrasp(OrcaResidualGrasp):
    def __init__(self, model, arm, reference, weights=None, config=None, speed_scale=.05):
        self.weights = dict(BASE_WEIGHTS if weights is None else weights)
        if set(self.weights) != set(BASE_WEIGHTS) or not all(math.isfinite(v) for v in self.weights.values()):
            raise ValueError('Unknown or missing reward components')
        if not math.isfinite(speed_scale) or speed_scale<=0:
            raise ValueError('Speed scale must be positive and finite')
        self.speed_scale=float(speed_scale)
        super().__init__(model, 'pure' if arm=='bc' else arm, reference, .08, config=config)

    def reset(self, rng):
        state=super().reset(rng)
        metrics=dict(state.metrics)
        for key in BASE_WEIGHTS:
            metrics['component_'+key]=jp.float32(0)
        metrics['cube_speed']=jp.float32(0)
        metrics['actuator_force_rms']=jp.float32(0)
        return state.replace(metrics=metrics)

    def step(self, state, action):
        nxt=super().step(state,action)
        m=nxt.metrics
        paired=(m['thumb_contact']>0)&(m['contact_count']>=2)&(m['forbidden_contact']==0)
        released=m['released']>0
        speed=jp.linalg.norm(self._cube_vel(nxt.data))
        components=dict(stable=m['stable'],done=nxt.done,
                        speed=jp.where(released&paired, jp.minimum((speed/self.speed_scale)**2,4),0.),
                        pair=(released&paired).astype(jp.float32),
                        hold_progress=m['stable']*jp.minimum(m['hold_sec']/30,1),
                        correction=jp.mean((nxt.info['requested_residual']/self.residual_bound)**2))
        reward=sum(self.weights[k]*v for k,v in components.items())*self.dt
        metrics=dict(m,**{'component_'+k:v for k,v in components.items()},cube_speed=speed,
                     actuator_force_rms=jp.sqrt(jp.mean(nxt.data.actuator_force**2)))
        return nxt.replace(reward=reward,metrics=metrics)
