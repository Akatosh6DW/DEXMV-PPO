"""Protocol coverage checks, without starting training."""

import unittest

from orca_formal_protocol import recipes, scenarios, SEEDS, STEPS


class FormalProtocolTests(unittest.TestCase):
    def test_complete_factorial_and_distinct_jobs(self):
        rows = recipes()
        self.assertEqual(len(rows), 80)
        self.assertEqual(len({r["label"] for r in rows}), len(rows))
        main = [r for r in rows if r["phase"] == "main"]
        self.assertEqual(len(main), 40)
        for seed in SEEDS:
            self.assertEqual({(r["method"], r["tactile"]) for r in main if r["seed"] == seed},
                             {(m, t) for m in ("bc", "ppo", "guided", "residual") for t in ("off", "on")})
        self.assertTrue(all(r["std"] == .7 for r in main))
        self.assertTrue(all(r["steps"] == (0 if r["method"] == "bc" else STEPS) for r in rows))

    def test_paired_bc_precedes_guided_and_preserves_noise(self):
        previous = {}
        for r in recipes():
            if r["bc"]:
                b = previous[r["bc"]]
                for k in ("seed", "tactile", "std"):
                    self.assertEqual(r[k], b[k])
                self.assertEqual(b["method"], "bc")
            previous[r["label"]] = r

    def test_robustness_changes_one_factor_at_a_time(self):
        self.assertEqual(len(scenarios()), 12)
        self.assertEqual(scenarios()["nominal"], {})
        self.assertTrue(all(len(v) <= 1 for v in scenarios().values()))


if __name__ == "__main__":
    unittest.main()
