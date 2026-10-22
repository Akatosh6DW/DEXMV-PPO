"""Failure classification must separate gates and ignore terminal padding."""

import unittest

import numpy as np

from diagnose_s09_failure_modes import episode_reasons


class FailureTests(unittest.TestCase):
    def trace(self):
        audit = np.zeros((5, 13))
        audit[:, 0:5] = .03
        audit[:, 9:12] = 1
        qpos = np.zeros((5, 7))
        qpos[:, 2] = .25
        return dict(audit=audit, active=np.array([1, 1, 1, 1, 0], dtype=bool),
            qpos=qpos, initial_penetration=np.zeros(5),
            success30=np.zeros(5), best_hold=np.array([0, .05, .05, .1, .1]))

    def test_height_is_not_mislabeled_as_contact_loss(self):
        trace = self.trace()
        trace["qpos"][2, 2] = .234
        trace["audit"][2, 11] = 0
        trace["audit"][4, 11] = 0
        trace["audit"][4, 5] = 1
        result = episode_reasons(trace, np.array([0, 0, .25, 1, 0, 0, 0]), 0)
        self.assertEqual(result["valid_streak_breaks"], 1)
        self.assertEqual(result["break_events_by_gate"]["height_loss_ge_15mm"], 1)
        self.assertEqual(result["break_events_by_gate"]["thumb_load_le_0_02n"], 0)
        self.assertFalse(result["ever_released_forbidden"])
        self.assertEqual(result["valid_reconstruction_disagreement_frames"], 0)

    def test_overlapping_causes_and_reconstruction_disagreements_are_visible(self):
        trace = self.trace()
        trace["audit"][1, [0, 5, 11]] = [0, 1, 0]
        trace["audit"][3, 11] = 0
        result = episode_reasons(trace, np.array([0, 0, .25, 1, 0, 0, 0]), 0)
        self.assertEqual(result["valid_streak_breaks"], 2)
        self.assertEqual(result["break_events_by_gate"]["thumb_load_le_0_02n"], 1)
        self.assertEqual(result["break_events_by_gate"]["forbidden_contact"], 1)
        self.assertEqual(result["unreconstructed_breaks"], 1)
        self.assertEqual(result["valid_reconstruction_disagreement_frames"], 1)

    def test_all_case_wrapper_preserves_primary_fields(self):
        from evaluate_s09_failure_quality import add_reasons
        one = self.trace()
        trace = {k:np.stack([v, v]) for k,v in one.items()}
        trace["diagnostic_initial_qpos"] = np.array([[0, 0, .25, 1, 0, 0, 0]] * 2)
        original = [dict(seed=1, success30=False), dict(seed=2, success30=False)]
        rows = add_reasons(trace, original, 0)
        self.assertEqual(len(rows), 2)
        for row, old in zip(rows, original):
            self.assertEqual({k:row[k] for k in old}, old)
            self.assertNotIn("failure_gates", old)
            self.assertEqual(row["failure_gates"]["valid_reconstruction_disagreement_frames"], 0)
        with self.assertRaises(ValueError):
            add_reasons(trace, [dict(success30=True), dict(success30=False)], 0)


if __name__ == "__main__":
    unittest.main()
