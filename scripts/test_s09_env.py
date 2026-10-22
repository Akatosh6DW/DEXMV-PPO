"""JAX/native state isolation, observation and residual-interface tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as jp
import numpy as np

from orca_s09_env import OrcaS09Grasp, cylinder_axis
from s09_suspension import SuspendedCylinder, body_quaternion, scene_xml


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name)/"model.xml"
        self.bank = Path(self.tmp.name)/"bank.npz"
        self.model.write_text(scene_xml())
        np.savez_compressed(self.bank, ctrl=np.zeros((3, 2, 17)), time=[0.,2.],
                            object_quaternion=[body_quaternion(a) for a in np.eye(3)])

    def test_reset_shape_and_continuous_orientation(self):
        env = OrcaS09Grasp(self.model)
        states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(901), 32))
        self.assertEqual(states.obs["state"].shape, (32,123))
        self.assertEqual(states.obs["privileged_state"].shape, (32,140))
        np.testing.assert_array_equal(states.obs["state"][:,94], 0)
        np.testing.assert_array_equal(states.obs["state"][:,95:112], 0)
        self.assertEqual(len(np.unique(np.asarray(states.data.qpos[:,env.oid+3:env.oid+7]), axis=0)), 32)
        self.assertLessEqual(float(states.info["initial_penetration"].max()), .001)

    def test_explicit_residual_and_reproducible_interleaved_states(self):
        env = OrcaS09Grasp(self.model, "residual", self.bank)
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        a = reset(jax.random.PRNGKey(1))
        b = reset(jax.random.PRNGKey(2))
        action = jp.ones(17)*.5
        first = step(a, action)
        _ = step(b, -action)
        second = step(a, action)
        np.testing.assert_array_equal(first.data.integration_words, second.data.integration_words)
        np.testing.assert_allclose(first.info["requested_residual"], .04, atol=1e-7)
        np.testing.assert_allclose(first.info["desired_ctrl"],
            jp.clip(first.info["video_base"]+.04, env._lowers, env._uppers), atol=1e-7)
        self.assertLessEqual(float(jp.max(jp.abs(first.info["effective_residual"]))), .080001)

    def test_axis_descriptor_matches_actual_cylinder(self):
        env = OrcaS09Grasp(self.model)
        state = env.reset(jax.random.PRNGKey(912))
        q = np.asarray(state.data.qpos[env.oid+3:env.oid+7])
        from scipy.spatial.transform import Rotation
        actual = -Rotation.from_quat(q[[1,2,3,0]]).as_matrix()[:,1]
        self.assertAlmostEqual(abs(np.dot(np.asarray(cylinder_axis(jp.array(q))), actual)), 1., places=6)

    def test_callback_matches_direct_native_steps(self):
        import mujoco
        from orca_native_physics import STATE_SPEC
        env = OrcaS09Grasp(self.model)
        state = jax.jit(env.reset)(jax.random.PRNGKey(903))
        sim = SuspendedCylinder(self.model.read_text())
        sim.reset(np.asarray(state.info["cube_spawn_pos"]),
                  np.asarray(state.data.qpos[env.oid+3:env.oid+7]))
        words = np.asarray(state.data.integration_words).copy().view(np.float64)
        mujoco.mj_setState(sim.model, sim.data, words, STATE_SPEC)
        step = jax.jit(env.step)
        for _ in range(65):
            action = env.ctrl_to_action(jp.zeros(17))
            state = step(state, action)
            row = sim.step(np.asarray(state.info["desired_ctrl"]))
            np.testing.assert_allclose(state.data.qpos, sim.data.qpos, atol=2e-6)
            np.testing.assert_allclose(state.data.audit[:5], row["loads"], atol=2e-5)
            np.testing.assert_allclose(state.data.audit[8], row["rope_force_n"], atol=2e-6)


if __name__ == "__main__":
    unittest.main()
