"""Focused checks without an expensive full physics compilation."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np
from flax import struct

from orca_refit_env import BASE_WEIGHTS, OrcaRefitGrasp
from orca_residual_env import OrcaResidualGrasp
from orca_guidance_recheck import task_reward
from train_orca_refit import check_gates, evaluator
from train_orca_residual import initialize_zero_mean
from test_orca_guidance_recheck import TerminalEnv

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/orca_collision_refit/G07_convex_palm_candidate/cube.xml'


@struct.dataclass
class FakeState:
    metrics: dict
    info: dict
    data: object
    done: object
    reward: object


class RefitTests(unittest.TestCase):
    def test_gate_and_mesh_hashes(self):
        check_gates(MODEL,ROOT/'artifacts/orca_collision_refit/G07_mjx_smoke/report.json')
        check_gates(ROOT/'artifacts/orca_collision_refit/G08_thumb_convex_candidate/cube.xml',
                    ROOT/'artifacts/orca_collision_refit/G08_mjx_smoke/report.json')

    def test_new_evaluator_excludes_post_terminal_rewards(self):
        class InstrumentedTerminalEnv(TerminalEnv):
            def reset(self,key):
                state=super().reset(key)
                metrics=dict(state.metrics,cube_speed=jp.float32(0),actuator_force_rms=jp.float32(0))
                metrics.update({'component_'+name:jp.float32(0) for name in BASE_WEIGHTS})
                return state._replace(metrics=metrics)
        net,params=initialize_zero_mean(101,.7)
        trace=jax.device_get(evaluator(InstrumentedTerminalEnv(),net,[9200])(params))
        self.assertEqual(float(trace['reward'].sum()),-7.)
        self.assertEqual(int(trace['active'].sum()),1)
        self.assertFalse(trace['alive'].any())

    def test_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):
            OrcaRefitGrasp(MODEL,'pure',None,{'typo':1})

    def test_millimeter_slip_scale_and_contact_gate(self):
        env=OrcaRefitGrasp(MODEL,'pure',None,dict(BASE_WEIGHTS,speed=-2),speed_scale=.005)
        env._cube_vel=lambda data:jp.array([.001,0,0])
        state=FakeState(dict(stable=jp.float32(1),thumb_contact=jp.float32(1),
                            contact_count=jp.float32(2),forbidden_contact=jp.float32(0),
                            released=jp.float32(1),hold_sec=jp.float32(15)),
                        dict(requested_residual=jp.zeros(17)),
                        SimpleNamespace(actuator_force=jp.ones(17)),jp.float32(0),jp.float32(0))
        with patch.object(OrcaResidualGrasp,'step',return_value=state):
            nxt=env.step(state,jp.zeros(17))
            self.assertAlmostEqual(float(nxt.metrics['component_speed']),.04,places=6)
            self.assertAlmostEqual(float(nxt.reward),(15-.08)*env.dt,places=6)
        for key,value in [('thumb_contact',0),('forbidden_contact',1),('released',0)]:
            altered=state.replace(metrics=dict(state.metrics,**{key:jp.float32(value)}))
            with patch.object(OrcaResidualGrasp,'step',return_value=altered):
                self.assertEqual(float(env.step(altered,jp.zeros(17)).metrics['component_speed']),0)

    def test_pure_and_bc_do_not_load_reference(self):
        with patch('orca_residual_env.load_reference',side_effect=AssertionError('reference leak')):
            for arm in ('pure','bc'):
                env=OrcaRefitGrasp(MODEL,arm,Path('/does/not/exist'))
                np.testing.assert_array_equal(env.reference_ctrl,0)

    def test_baseline_and_component_rewards(self):
        env=OrcaRefitGrasp(MODEL,'pure',None)
        env._cube_vel=lambda data:jp.array([.1,0,0])
        for stable,done in ((0,0),(1,0),(0,1),(1,1)):
            state=FakeState(dict(stable=jp.float32(stable),thumb_contact=jp.float32(1),
                                 contact_count=jp.float32(2),forbidden_contact=jp.float32(0),
                                 released=jp.float32(1),hold_sec=jp.float32(15)),
                            dict(requested_residual=jp.ones(17)*.04),
                            SimpleNamespace(actuator_force=jp.ones(17)*2),jp.float32(done),jp.float32(0))
            with patch.object(OrcaResidualGrasp,'step',return_value=state):
                nxt=env.step(state,jp.zeros(17))
                self.assertAlmostEqual(float(nxt.reward),float(task_reward(stable,done,env.dt)),places=6)
                self.assertEqual(float(nxt.metrics['component_speed']),4)
                self.assertAlmostEqual(float(nxt.metrics['component_correction']),.25)
                env.weights=dict(BASE_WEIGHTS,speed=-2.)
                modified=env.step(state,jp.zeros(17))
                self.assertAlmostEqual(float(modified.reward),float(nxt.reward)-8*env.dt,places=5)
                self.assertEqual(float(modified.done),done)
                self.assertEqual(float(modified.metrics['hold_sec']),15)
                env.weights=dict(BASE_WEIGHTS)


if __name__=='__main__':
    unittest.main()
