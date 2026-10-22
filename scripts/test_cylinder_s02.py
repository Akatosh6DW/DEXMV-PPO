"""Confirm that changing the object preserves the frozen S02 learning task."""

import io
import json
import pickle
import unittest

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from orca_cylinder_s02_env import OrcaCylinderS02Grasp, ROOT, SOURCE_MODEL
from orca_suite_env import OrcaSuiteGrasp, BENCHMARKS
from prepare_cylinder_s02_source import ArrayUnpickler, digest
from train_orca_acquisition import initial_gate

INPUTS = ROOT/'artifacts/orca_cylinder_s02/inputs'


class CylinderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update('jax_pmap_shmap_merge', False)

    def env(self, arm='residual', **kwargs):
        return OrcaCylinderS02Grasp(INPUTS/'model.xml', arm, INPUTS/'reference.npz',
                                   raw_reference=INPUTS/'unrefined.npz', **kwargs)

    def test_frozen_original(self):
        protocol = json.loads((ROOT/'artifacts/orca_benchmark_suite/S02/protocol.json').read_text())
        for path, expected in protocol['source_hashes'].items():
            self.assertEqual(digest(ROOT/path), expected, path)

    def test_only_shape_and_required_inertia_change(self):
        env = self.env()
        old = OrcaSuiteGrasp(SOURCE_MODEL, 'residual', INPUTS/'reference.npz')
        current, previous = json.loads(env._config.to_json()), json.loads(old._config.to_json())
        np.testing.assert_allclose(current['spawn_config']['support_pos'],
                                   previous['spawn_config']['support_pos'], atol=1e-9)
        current['spawn_config']['support_pos'] = previous['spawn_config']['support_pos']
        self.assertEqual(current, previous)
        self.assertEqual(env.mj_model.geom('cube').type[0], mujoco.mjtGeom.mjGEOM_CYLINDER)
        np.testing.assert_allclose(env.mj_model.geom('cube').size[:2], [.0265, .0265])
        self.assertAlmostEqual(env.mj_model.body('cube').mass[0], .035)
        np.testing.assert_array_equal(env.reference_ctrl, old.reference_ctrl)
        self.assertEqual(env.dt, .05)
        self.assertEqual(env.n_substeps, 10)

    def test_benchmark_and_initial_state(self):
        base = self.env()
        for name in BENCHMARKS:
            env = self.env(benchmark=name)
            self.assertTrue(initial_gate(env, 16)['passed'])
            state = jax.jit(env.reset)(jax.random.PRNGKey(333))
            self.assertEqual(state.obs['state'].shape, (123,))
            self.assertEqual(state.obs['privileged_state'].shape, (140,))
            if name == 'B3_friction':
                mask = env.mj_model.geom_contype != 0
                np.testing.assert_allclose(env.mj_model.geom_friction[mask, 0],
                                           base.mj_model.geom_friction[mask, 0]*.6)
            if name == 'B4_mass':
                self.assertAlmostEqual(env.mj_model.body('cube').mass[0], .07)
                np.testing.assert_allclose(env.mj_model.body('cube').inertia,
                                           base.mj_model.body('cube').inertia*2)

    def test_execution_is_reference_plus_residual(self):
        env = self.env()
        state = jax.jit(env.reset)(jax.random.PRNGKey(123))
        step = jax.jit(env.step)
        zero = step(state, jp.zeros(17))
        one = step(state, jp.ones(17)*.1)
        np.testing.assert_array_equal(zero.info['requested_residual'], np.zeros(17))
        self.assertGreater(np.linalg.norm(one.info['requested_residual']), 0)
        np.testing.assert_allclose(one.info['desired_ctrl'],
            np.clip(one.info['video_base']+one.info['requested_residual'], env._lowers, env._uppers), atol=1e-6)
        off = self.env(residual_gain=0.)
        state = jax.jit(off.reset)(jax.random.PRNGKey(123))
        nxt = jax.jit(off.step)(state, jp.ones(17))
        np.testing.assert_array_equal(nxt.info['requested_residual'], np.zeros(17))

    def test_untrusted_globals_rejected(self):
        with self.assertRaises(pickle.UnpicklingError):
            ArrayUnpickler(io.BytesIO(pickle.dumps(eval))).load()


if __name__ == '__main__':
    unittest.main()
