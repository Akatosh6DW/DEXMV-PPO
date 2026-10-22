"""The residual audit checks executed actions, and still rejects bad padding."""

import unittest

import numpy as np

from audit_s09_replay import check_residual_composition


class AuditTests(unittest.TestCase):
    def trace(self):
        active = np.array([[True, True, False, False]])
        data = {key: np.zeros((1, 4, 17)) for key in
            ("action", "video_base", "desired_ctrl", "requested_residual", "effective_residual",
             "qpos", "qvel", "ctrl", "audit")}
        data["active"] = active
        data["action"][0, 2:] = .5
        return data

    def test_unused_action_after_terminal_is_not_executed(self):
        self.assertEqual(check_residual_composition(self.trace(), 0, np.tile([-1., 1.], (17, 1))), 0)

    def test_wrong_executed_action_is_rejected(self):
        data = self.trace()
        data["action"][0, 1] = .5
        with self.assertRaises(AssertionError):
            check_residual_composition(data, 0, np.tile([-1., 1.], (17, 1)))

    def test_changed_frozen_state_is_rejected(self):
        data = self.trace()
        data["ctrl"][0, 2] = .1
        with self.assertRaises(AssertionError):
            check_residual_composition(data, 0, np.tile([-1., 1.], (17, 1)))


if __name__ == "__main__":
    unittest.main()
