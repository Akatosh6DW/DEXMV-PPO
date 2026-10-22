"""Retention policy tests; these tests never delete files."""

import unittest

from prune_legacy_aero_checkpoints import preserved_steps


class RetentionTests(unittest.TestCase):
    def test_endpoints_landmarks_and_references(self):
        kept = preserved_steps(range(0, 101, 10), {}, {30})
        self.assertEqual(set(kept), {0, 20, 30, 50, 80, 100})
        self.assertIn("referenced_step", kept[30])

    def test_unsaved_extrema_keep_both_neighbors(self):
        kept = preserved_steps(range(0, 101, 10), {
            "eval/episode_reward": [(35, -10), (65, 10)],
            "eval/episode_diagnostic/contact": [(15, 0), (95, 1)],
        }, set())
        self.assertTrue({30, 40, 60, 70, 10, 20, 90, 100}.issubset(kept))

    def test_single_checkpoint(self):
        kept = preserved_steps([10], {"eval/episode_reward": [(0, 0), (20, 1)]}, set())
        self.assertEqual(set(kept), {10})


if __name__ == "__main__":
    unittest.main()
