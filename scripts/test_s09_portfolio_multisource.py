"""Keep the historical-selection inputs separate from confirmation labels."""

import unittest

from confirm_s09_portfolio_multisource import COUNT, GROUPS, START, validate_selection_rows


class MultiSourcePortfolioTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(seed=130000+i, selected_reference=i % 6, success30=i % 2 == 0) for i in range(64)]

    def test_original_development_only(self):
        self.assertEqual(sum(validate_selection_rows(self.rows)), 32)
        for seed in (190000, START):
            rows = [dict(r, seed=seed+i) for i, r in enumerate(self.rows)]
            with self.assertRaises(ValueError):
                validate_selection_rows(rows)

    def test_invalid_labels_or_order(self):
        for rows in (self.rows[:-1], self.rows[::-1],
                     [dict(r, selected_reference=6) for r in self.rows],
                     [dict(r, success30=1) for r in self.rows]):
            with self.assertRaises(ValueError):
                validate_selection_rows(rows)

    def test_include_stronger_stage_control(self):
        self.assertEqual(GROUPS, ("bc", "stage", "expert_last", "portfolio"))
        self.assertEqual((START, COUNT), (200000, 256))


if __name__ == "__main__":
    unittest.main()
