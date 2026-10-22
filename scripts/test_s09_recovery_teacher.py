"""Recovery probe must preserve physics and never count a short hold as 30s."""

import unittest
import numpy as np

from probe_s09_recovery_teacher import capture, restore, rollout, offset_controls, objective_value, unpack_counters
from s09_suspension import SuspendedCylinder, body_quaternion


class RecoveryTeacherTests(unittest.TestCase):
    def test_serialized_counters_restore_original_types(self):
        result = unpack_counters([.08,.055,.25], [0., 4., 7., 1., .001])
        self.assertIs(type(result["hold"]), int)
        self.assertIs(type(result["ever_forbidden"]), bool)
        self.assertTrue(result["ever_forbidden"])
        with self.assertRaises(ValueError):
            unpack_counters([0,0,0], [0,4.5,7,1,0])
        with self.assertRaises(ValueError):
            unpack_counters([0,0,0], [0,4,7,2,0])

    def test_restore_is_exact_including_counters_and_warmstart(self):
        sim = SuspendedCylinder()
        sim.reset([.08,.055,.25], body_quaternion([0,0,1]))
        for _ in range(3):
            sim.step(np.zeros(17))
        saved = capture(sim)
        result, trace = rollout(sim, saved, np.zeros((5,17)))
        restored, other = rollout(sim, saved, np.zeros((5,17)))
        self.assertEqual(result, restored)
        self.assertNotIn("success30", result)
        self.assertFalse(result["recovery_candidate"])
        for key in trace:
            np.testing.assert_array_equal(trace[key], other[key])
        restore(sim, saved)
        np.testing.assert_array_equal(capture(sim)[0], saved[0])

    def test_offsets_leave_unselected_joints_and_input_unchanged(self):
        base = np.zeros((4,17))
        limits = np.tile([-.1,.1],(17,1))
        result = offset_controls(base, np.array([.2,-.05]), [13,14], limits)
        np.testing.assert_array_equal(base, 0)
        np.testing.assert_array_equal(result[:,13], .1)
        np.testing.assert_array_equal(result[:,14], -.05)
        np.testing.assert_array_equal(result[:,:13], 0)
        with self.assertRaises(ValueError):
            offset_controls(base, np.zeros(1), [13,14], limits)

    def test_cost_penalizes_dead_or_forbidden_candidates(self):
        row = dict(alive_at_end=True, forbidden=False, max_penetration_m=.002,
            mean_positive_height_loss_m=0., mean_coverage=1., valid_fraction=1.)
        zero = np.zeros(10)
        self.assertEqual(objective_value(row, zero), 0)
        self.assertGreater(objective_value(dict(row, alive_at_end=False), zero), 99)
        self.assertGreater(objective_value(dict(row, forbidden=True), zero), 99)


if __name__ == "__main__":
    unittest.main()
