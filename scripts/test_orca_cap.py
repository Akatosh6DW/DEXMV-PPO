"""Native structural and negative-control tests for the cap scene."""

from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from orca_cap_scene import build_scene, indices, native_rollout, PITCH
from orca_cap_metrics import hold_seconds


class CapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = build_scene(Path(self.tmp.name) / "scene.xml")

    def test_only_hand_actuated_and_bottle_fixed(self):
        m = self.model
        self.assertEqual(m.nu, 17)
        self.assertEqual(m.body("fixed_bottle").jntnum[0], 0)
        cap_joints = {m.joint("cap_turn").id, m.joint("cap_lift").id}
        self.assertFalse(cap_joints & set(m.actuator_trnid[:, 0]))
        self.assertEqual(m.neq, 1)
        self.assertAlmostEqual(m.eq_data[0, 1], 1/PITCH, places=5)

    def test_no_input_does_not_turn_cap(self):
        ctrl = np.zeros((1, 17))
        ctrl[0, 0] = -.25
        trace, report = native_rollout(self.model, ctrl, seconds=2)
        self.assertLess(abs(report["max_degrees"]), 1e-5)
        self.assertLess(report["max_thread_error_m"], 1e-8)
        self.assertTrue(np.isfinite(trace["qpos"]).all())

    def test_screw_kinematic_relation(self):
        m = self.model
        d = mujoco.MjData(m)
        _, turn, lift, _ = indices(m)
        d.qpos[turn], d.qpos[lift] = np.pi/2, .0005
        mujoco.mj_forward(m, d)
        self.assertLess(abs(d.efc_pos[0]), 1e-7)
        self.assertAlmostEqual(d.qpos[lift], PITCH*d.qpos[turn])


class MetricTests(unittest.TestCase):
    def trace(self):
        return dict(angle=np.ones((1, 30))*2, paired=np.ones((1, 30)),
                    thread_error=np.zeros((1, 30)), forbidden=np.zeros((1, 30)),
                    alive=np.ones((1, 30), dtype=bool))

    def test_clean_success(self):
        self.assertEqual(hold_seconds(self.trace())[0], 1.5)

    def test_late_violation_cancels_previous_success(self):
        for name, value in (("thread_error", .002), ("forbidden", 1), ("alive", False)):
            trace = self.trace()
            trace[name][0, -1] = value
            self.assertEqual(hold_seconds(trace)[0], 0.)

    def test_contact_gap_resets_continuous_hold(self):
        trace = self.trace()
        trace["paired"][0, 15] = 0
        self.assertEqual(hold_seconds(trace)[0], .75)


if __name__ == "__main__":
    unittest.main()
