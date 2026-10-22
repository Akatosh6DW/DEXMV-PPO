"""Geometry, physical movement and no-object-attachment checks for S10."""

import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from s10_mobile_base import CENTER, MobileCylinder, align_rotation, quaternion, rotation


class MobileBaseTests(unittest.TestCase):
    def test_axis_alignment_and_symmetry(self):
        rng = np.random.default_rng(73)
        for _ in range(50):
            source, target = [quaternion(Rotation.random(random_state=rng)) for _ in range(2)]
            for rule in ("minimal", "gravity"):
                rot = align_rotation(source, target, rule)
                a = rot.apply(rotation(source).apply([0., -1., 0.]))
                b = rotation(target).apply([0., -1., 0.])
                self.assertAlmostEqual(abs(a@b), 1., places=12)

    def test_real_free_joint_no_object_weld(self):
        sim = MobileCylinder()
        self.assertEqual(sim.model.joint("s10_hand_free").type, mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(sim.model.nu, 17)
        self.assertNotIn(sim.body, sim.model.eq_obj1id)
        self.assertNotIn(sim.body, sim.model.eq_obj2id)
        self.assertEqual(sim.model.nq, 31)

    def test_common_home_and_pose_composition(self):
        sim = MobileCylinder()
        homes = []
        source_object = sim.bank["object_position"][0]+[0, 0, .25]
        for quat in (np.array([1., 0., 0., 0.]), quaternion(Rotation.from_rotvec([1., .3, -.4]))):
            sim.reset(CENTER, quat)
            homes.append(sim.data.qpos[sim.root_qid:sim.root_qid+7].copy())
            local = sim.original_rot.inv().apply(source_object-sim.original_pos)
            np.testing.assert_allclose(sim.goal_pos+sim.goal_rot.apply(local), CENTER, atol=1e-12)
            self.assertEqual(sim.initial_penetration, 0.)
        np.testing.assert_array_equal(*homes)

    def test_target_does_not_teleport_physics(self):
        sim = MobileCylinder()
        sim.reset(CENTER, quaternion(Rotation.from_rotvec([.5, -.2, .8])))
        initial = sim.data.qpos.copy()
        sim.target(2.)
        np.testing.assert_array_equal(sim.data.qpos, initial)
        sim.step(np.zeros(17))
        self.assertGreater(np.linalg.norm(sim.data.qpos[:3]-initial[:3]), 1e-7)
        self.assertGreater(np.linalg.norm(sim.data.qpos[:3]-sim.goal_pos), .01)

    def test_release_no_rope_force(self):
        sim = MobileCylinder()
        sim.reset(CENTER, sim.bank["object_quaternion"][0])
        for _ in range(110):
            row = sim.step(sim.reference(sim.data.time+.05))
        self.assertTrue(row["released"])
        self.assertEqual(row["rope_force_n"], 0.)
        self.assertLess(row["base_position_error_m"], .001)
        self.assertTrue(np.isfinite(sim.data.qpos).all())


if __name__ == "__main__":
    unittest.main()
