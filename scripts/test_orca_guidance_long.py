"""Tests for fixed evaluations, paired summaries and non-fabricated hit times."""

import unittest

import numpy as np

from evaluate_orca_guidance_long import aggregate, condition_config
from report_orca_guidance_long import bootstrap_band, confirmed_threshold


class LongStudyTests(unittest.TestCase):
    def test_single_hit_is_censored(self):
        curve = [dict(step=i*100, success30=v) for i, v in enumerate([0, .8, .7, .9])]
        self.assertIsNone(confirmed_threshold(curve, .75))

    def test_confirmation_uses_second_checkpoint(self):
        curve = [dict(step=i*100, success30=v) for i, v in enumerate([.8, .8, .1])]
        self.assertEqual(confirmed_threshold(curve, .75), 100)

    def test_bootstrap_resamples_seeds_and_preserves_constants(self):
        values = np.tile([.2, .4, .6], (5, 1))
        for array in bootstrap_band(values):
            np.testing.assert_allclose(array, [.2, .4, .6])
        with self.assertRaises(ValueError):
            bootstrap_band([[np.nan, 1], [0, 0]])

    def test_test_sets_are_disjoint_and_budgets_fixed(self):
        nominal, offset = condition_config("nominal"), condition_config("offset")
        self.assertFalse(set(nominal["seeds"]) & set(offset["seeds"]))
        self.assertEqual(len(nominal["seeds"]), 64)
        self.assertEqual(nominal["checkpoints"]["short"], 655360)
        self.assertEqual(offset["jitter"], 4*nominal["jitter"])

    def test_aggregate_pairs_by_seed_and_rejects_missing(self):
        protocol = dict(seeds=[20, 21], arms=["pure", "bc", "residual"], rewards=["shaped", "hold_only"])
        rows = [dict(seed=seed, arm=arm, reward=reward, label=label, success30=score)
                for reward in protocol["rewards"] for arm in protocol["arms"]
                for seed in protocol["seeds"] for label, score in [("initial", .25), ("terminal", .5)]]
        result = aggregate(rows[::-1], protocol, ["initial", "terminal"])
        self.assertEqual(result["groups"]["shaped_bc_gain_from_initial"]["mean"], .25)
        self.assertEqual(result["paired"]["shaped_bc_minus_pure_terminal"]["mean"], 0)
        with self.assertRaises(ValueError):
            aggregate(rows[:-1], protocol, ["initial", "terminal"])


if __name__ == "__main__":
    unittest.main()
