"""Observation, physics and initialization contracts; run with JAX_PLATFORMS=cpu."""

from pathlib import Path
import unittest

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from orca_residual_env import OrcaResidualGrasp
from train_orca_residual import initialize_zero_mean

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def test_object_geometry_keeps_hand_mass_friction_and_actuators(self):
        models = [mujoco.MjModel.from_xml_path(str(ROOT/f"artifacts/orca_residual/models/{s}.xml"))
                  for s in ("cube", "sphere", "cuboid")]
        original = models[0]
        for model in models:
            self.assertEqual((model.nq, model.nv, model.nu), (original.nq, original.nv, 17))
            self.assertAlmostEqual(float(model.body("cube").mass[0]), .035)
            np.testing.assert_array_equal(model.body_mass, original.body_mass)
            np.testing.assert_array_equal(model.actuator_ctrlrange, original.actuator_ctrlrange)
            np.testing.assert_array_equal(model.actuator_gainprm, original.actuator_gainprm)
            np.testing.assert_array_equal(model.geom_friction, original.geom_friction)
            hand = np.arange(model.nbody) != model.body("cube").id
            np.testing.assert_array_equal(model.body_inertia[hand], original.body_inertia[hand])
            self.assertTrue(np.all(model.geom_condim == 4))

    def test_pure_never_loads_demo_and_shapes_match(self):
        env = OrcaResidualGrasp(ROOT/"artifacts/orca_residual/models/sphere.xml", "pure",
                               reference="/nonexistent_reference_must_not_be_read.npz")
        self.assertAlmostEqual(env.half_height, .021)
        self.assertEqual(env.action_size, 17)
        obs = jax.eval_shape(env.reset, jax.random.PRNGKey(0)).obs
        self.assertEqual(obs["state"].shape, (113,))
        self.assertEqual(obs["privileged_state"].shape, (130,))
        self.assertEqual(float(jp.max(jp.abs(env.reference_ctrl))), 0)

    def test_matched_zero_means_and_exploration(self):
        net, params = initialize_zero_mean(90, .7)
        logits = net.policy_network.apply(params[0], params[1],
                                          {"state": jp.linspace(-3, 3, 113)})
        np.testing.assert_array_equal(logits[:17], np.zeros(17))
        np.testing.assert_allclose(jax.nn.softplus(logits[17:])+.001, .7, atol=1e-6)
        _, other = initialize_zero_mean(90, .7)
        for x, y in zip(jax.tree.leaves(params), jax.tree.leaves(other)):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
