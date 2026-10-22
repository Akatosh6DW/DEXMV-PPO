"""Cap release must never disable the independent hand positioner."""

import unittest

import mujoco
import numpy as np

from s11_cap import CapPilot


class CapTests(unittest.TestCase):
    def test_free_root_and_passive_cap(self):
        sim=CapPilot()
        self.assertEqual(sim.model.joint("s11_hand_free").type,mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(sim.model.nu,17)
        for joint in sim.model.actuator_trnid[:,0]:
            self.assertFalse(sim.model.joint(joint).name.startswith("cap_"))

    def test_thread_release_keeps_positioner(self):
        sim=CapPilot();sim.reset(240001)
        sim.data.qpos[sim.turn]=sim.release_angle+.01
        self.assertFalse(sim.thread_mode())
        self.assertFalse(sim.data.eq_active[sim.thread_ids].any())
        self.assertTrue(sim.data.eq_active[sim.hand_eq])
        sim.data.qpos[sim.turn]=0
        self.assertFalse(sim.thread_mode())

    def test_target_does_not_teleport(self):
        sim=CapPilot();sim.reset(240001)
        initial=sim.data.qpos.copy()
        target,_=sim.target(9.)
        np.testing.assert_array_equal(sim.data.qpos,initial)
        self.assertGreater(np.linalg.norm(target-sim.home_pos),.01)

    def test_fixed_controller_and_reset(self):
        sim=CapPilot(moving=False);sim.reset(240001)
        for t in (0,10,24):
            np.testing.assert_array_equal(sim.target(t)[0],sim.home_pos)
        sim.data.eq_active[sim.thread_ids]=False
        sim.reset(240001)
        self.assertTrue(sim.data.eq_active.all())

    def test_root_offset_is_reset_only_and_not_cumulative(self):
        original=CapPilot();original.reset(240001)
        offset=np.array([.001,-.002,.003])
        sim=CapPilot(root_offset=offset);sim.reset(240001)
        np.testing.assert_allclose(sim.home_pos,original.home_pos+offset)
        np.testing.assert_allclose(sim.center,original.center)
        sim.reset(240001)
        np.testing.assert_allclose(sim.home_pos,original.home_pos+offset)
        np.testing.assert_allclose(sim.data.mocap_pos[sim.target_id],sim.home_pos)


if __name__=="__main__":
    unittest.main()
