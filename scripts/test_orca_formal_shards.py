"""Evaluation sharding must preserve every preregistered job exactly once."""

import unittest

from run_orca_formal_evaluations import jobs, shard_jobs


class FormalShardTests(unittest.TestCase):
    def test_two_workers_preserve_full_design_without_duplicates(self):
        for phase in ("main", "reference", "exploration", "all"):
            first, second = shard_jobs(phase, 0, 2), shard_jobs(phase, 1, 2)
            labels = [r["label"] for r in first + second]
            self.assertEqual(len(labels), len(set(labels)))
            self.assertEqual(set(labels), {r["label"] for r in jobs(phase)})
        self.assertEqual(len(jobs("all")), 513)

    def test_invalid_shard_fails(self):
        for index, count in ((-1, 2), (2, 2), (0, 0)):
            with self.assertRaises(ValueError):
                shard_jobs("main", index, count)


if __name__ == "__main__":
    unittest.main()
