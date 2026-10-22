"""The diagnostic intervention must change only the declared input rows."""

import copy
import unittest

import jax.numpy as jp

from make_s09_feedback_ablation import verify_only_feedback_removed
from run_s09_bc_phase_curriculum import KEEP, phase_initialization


class FeedbackAblationTests(unittest.TestCase):
    def sample(self):
        return (None, {"params": {
            "hidden_0": {"kernel": jp.ones((123, 2)), "bias": jp.ones(2)},
            "hidden_2": {"kernel": jp.ones((2, 34)), "bias": jp.ones(34)}}},
            {"critic": jp.ones(3)})

    def test_only_declared_rows_change(self):
        before = self.sample()
        after = phase_initialization(before)
        self.assertEqual(verify_only_feedback_removed(before, after), int((~KEEP).sum()*2))
        self.assertEqual(float(before[1]["params"]["hidden_0"]["kernel"][0,0]), 1.)

    def test_other_parameter_change_rejected(self):
        before = self.sample()
        after = copy.deepcopy(phase_initialization(before))
        after[2]["critic"] = jp.zeros(3)
        with self.assertRaises(AssertionError):
            verify_only_feedback_removed(before, after)


if __name__ == "__main__":
    unittest.main()
