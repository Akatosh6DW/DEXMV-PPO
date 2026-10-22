"""Physical negative controls for the S09 string model and pose sampler."""

import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from s09_suspension import (SuspendedCylinder, body_quaternion,
                            orientation_samples, rope_force)


class SuspensionTests(unittest.TestCase):
    def test_rope_is_unilateral_and_releases(self):
        np.testing.assert_array_equal(rope_force(np.zeros(3), np.zeros(3), [0, 0, .19], 0), 0)
        np.testing.assert_allclose(rope_force(np.zeros(3), np.zeros(3), [0, 0, .21], 0), [0, 0, 1])
        np.testing.assert_array_equal(rope_force(np.zeros(3), [0, 0, 10], [0, 0, .21], 0), 0)
        np.testing.assert_array_equal(rope_force(np.zeros(3), np.zeros(3), [0, 0, .5], 3), 0)

    def test_orientation_sampler_and_geom_axis(self):
        q = orientation_samples(4096, 9001)
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-12)
        axes = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()[:, :, 1]
        np.testing.assert_allclose(axes.mean(0), 0, atol=.04)
        np.testing.assert_allclose(np.mean(axes**2, axis=0), 1/3, atol=.04)
        sim = SuspendedCylinder()
        for axis in np.eye(3):
            sim.reset([.08, .08, .25], body_quaternion(axis))
            np.testing.assert_allclose(sim.data.geom_xmat[sim.model.geom("cube").id].reshape(3, 3)[:, 2], axis, atol=1e-12)

    def test_no_hand_cannot_hold_after_release(self):
        sim = SuspendedCylinder()
        mask = sim.model.geom_contype == 1
        sim.model.geom_contype[mask] = sim.model.geom_conaffinity[mask] = 0
        self.assertEqual(sim.model.neq, 0)
        np.testing.assert_array_equal(sim.model.tendon_stiffness, 0)
        for quaternion in orientation_samples(3, 9002, include_axes=True):
            sim.reset([.08, .08, .30], quaternion)
            for _ in range(60):
                row = sim.step(np.zeros(17))
                self.assertFalse(row["released"])
                self.assertAlmostEqual(sim.data.qpos[sim.oid+2], .30, places=7)
                self.assertEqual(row["hold_sec"], 0)
            for _ in range(8):
                row = sim.step(np.zeros(17))
                self.assertEqual(row["rope_force_n"], 0)
            self.assertFalse(row["alive"])
            self.assertFalse(row["success30"])
            self.assertEqual(row["hold_sec"], 0)

    def test_complete_reset_has_no_prior_episode_dependency(self):
        sim = SuspendedCylinder()
        q = body_quaternion([0, 0, 1])
        sim.reset([.08, .08, .3], q)
        initial = sim.data.qpos.copy()
        sim.step(np.zeros(17))
        sim.reset([.08, .08, .3], q)
        np.testing.assert_array_equal(sim.data.qpos, initial)
        self.assertEqual(sim.data.time, 0)
        self.assertEqual(sim.hold, 0)
        np.testing.assert_array_equal(sim.data.xfrc_applied, 0)


if __name__ == "__main__":
    unittest.main()
