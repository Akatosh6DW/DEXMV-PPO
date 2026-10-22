"""Sensitivity analysis must retain raw failure and forbidden-contact gates."""

import unittest

from supplement_orca_s03_quality import describe, boolean


class QualityTests(unittest.TestCase):
    def test_threshold_only_changes_penetration_gate(self):
        def row(raw, forbidden, depth):
            return dict(success30=str(raw), substep_forbidden=str(forbidden),
                        clean_success30=str(raw and not forbidden and depth <= .003),
                        post_release_penetration_m=str(depth), substep_force_peak_n="10")
        curves, summary = describe([row(True, False, .0033), row(True, False, .0015),
                                    row(True, True, .001), row(False, False, 0)])
        self.assertEqual([c["success_rate"] for c in curves], [.25, .25, .5, .5])
        self.assertEqual(summary["raw_success_count"], 3)
        self.assertEqual(summary["clean_success_count"], 1)
        self.assertEqual(summary["force_clean_success_mean_n"], 10)

    def test_no_success_is_not_zero_force(self):
        rows = [dict(success30="False", substep_forbidden="False", clean_success30="False",
                     post_release_penetration_m="0", substep_force_peak_n="2")]
        _, quality = describe(rows)
        self.assertIsNone(quality["force_raw_success_mean_n"])
        self.assertIsNone(quality["force_clean_success_mean_n"])
        with self.assertRaises(ValueError):
            boolean("")


if __name__ == "__main__":
    unittest.main()
