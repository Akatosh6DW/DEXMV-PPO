import unittest

from analyze_orca_s02_s04 import boolean, matrix, outcomes, score


class AnalysisTests(unittest.TestCase):
    def test_strict_boolean_parsing(self):
        self.assertFalse(boolean("False"))
        self.assertTrue(boolean("True"))
        with self.assertRaises(ValueError):
            boolean("0")

    def test_score_requires_all_gates(self):
        self.assertTrue(score(True, False, .003))
        self.assertFalse(score(True, False, .00301))
        self.assertFalse(score(False, False, 0))
        self.assertFalse(score(True, True, 0))

    def test_paired_matrix_rejects_mismatched_or_duplicate_cases(self):
        rows = [dict(train_seed=1, case=5, clean_success30=True),
                dict(train_seed=2, case=6, clean_success30=False)]
        with self.assertRaises(ValueError):
            matrix(rows)
        with self.assertRaises(ValueError):
            matrix([rows[0], rows[0]])
        rows[1]["case"] = 5
        a, seeds, cases = matrix(rows)
        self.assertEqual(a.shape, (2, 1))
        self.assertEqual(seeds, [1, 2])
        self.assertEqual(cases, [5])

    def test_mutually_exclusive_outcomes(self):
        rows = [dict(success30=True, clean_success30=True),
                dict(success30=True, clean_success30=False),
                dict(success30=False, clean_success30=False)]
        self.assertEqual(outcomes(rows), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
