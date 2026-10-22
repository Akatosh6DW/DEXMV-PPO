"""Known perturbation timing and interpretation bounds for feedback diagnostics."""

import unittest

import numpy as np

from audit_orca_feedback_response import compare


class ResponseTests(unittest.TestCase):
    def traces(self):
        trace = dict(qpos=np.zeros((1, 800, 24)), policy_action=np.zeros((1, 800, 17)),
                     desired_ctrl=np.zeros((1, 800, 17)), requested_residual=np.zeros((1, 800, 17)),
                     alive=np.ones((1, 800), bool), active=np.ones((1, 800), bool), hold=np.ones((1, 800)))
        return trace, {k: v.copy() for k, v in trace.items()}

    def test_fixed_reference(self):
        a, b = self.traces()
        b["qpos"][:, 200:] = .01
        row = compare(a, b)[0]
        self.assertTrue(row["matched_prefix"])
        self.assertFalse(row["responding"])

    def test_closed_loop_response(self):
        a, b = self.traces()
        b["qpos"][:, 200:] = .01
        b["policy_action"][:, 201:] = .1
        b["desired_ctrl"][:, 201:] = .008
        b["requested_residual"][:, 201:] = .008
        row = compare(a, b)[0]
        self.assertTrue(row["responding"])
        self.assertAlmostEqual(row["residual_response_rms_rad"], .008)

    def test_unmatched_prefix_not_causal(self):
        a, b = self.traces()
        b["qpos"][:, 100] = .01
        b["desired_ctrl"][:, 201:] = .008
        row = compare(a, b)[0]
        self.assertFalse(row["matched_prefix"])
        self.assertFalse(row["responding"])

    def test_no_grasp_no_recovery_claim(self):
        a, b = self.traces()
        b["hold"][:, 199] = 0
        b["desired_ctrl"][:, 201:] = .008
        self.assertFalse(compare(a, b)[0]["responding"])


if __name__ == "__main__":
    unittest.main()
