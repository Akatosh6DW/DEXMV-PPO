"""Paired initial-state statistics do not invent independent training runs."""

import unittest
import numpy as np
from analyze_s09_retention_check import check_source_hashes, paired_result


class RetentionCheckTests(unittest.TestCase):
    def test_shared_source_evidence_is_separate(self):
        self.assertEqual(check_source_hashes({"a.py": "a", "b.py": "b"},
            {"scripts/a.py": "a"}, [{"b.py": "b"}, {"b.py": "b"}]), {"b.py": "b"})

    def test_reject_source_drift(self):
        for frozen, historical in (({"scripts/a.py": "old"}, []),
                                   ({}, [{"a.py": "a"}, {"a.py": "old"}])):
            with self.assertRaises(ValueError):
                check_source_hashes({"a.py": "a"}, frozen, historical)

    def test_reject_unproven_shared_source(self):
        for historical in ([], [{}], [{"a.py": "a"}, {}]):
            with self.assertRaises(ValueError):
                check_source_hashes({"a.py": "a"}, {}, historical)
        with self.assertRaises(ValueError):
            check_source_hashes({"../a.py": "a"}, {}, [{"../a.py": "a"}])

    def test_counts_and_reproducibility(self):
        a, b = np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
        result = paired_result(a, b, draws=200)
        self.assertEqual(result, paired_result(a, b, draws=200))
        for key in ("gained", "lost", "both_success", "both_failure"):
            self.assertEqual(result[key], 1)
        self.assertEqual(result["gain_pp"], 0)

    def test_identical_labels(self):
        result = paired_result([1, 0], [1, 0], draws=200)
        self.assertEqual(result["ci95_low_pp"], 0)
        self.assertEqual(result["ci95_high_pp"], 0)

    def test_reject_unpaired_or_nonbinary(self):
        for a, b in (([], []), ([1], [1, 0]), ([2], [0])):
            with self.assertRaises(ValueError):
                paired_result(a, b)


if __name__ == "__main__":
    unittest.main()
