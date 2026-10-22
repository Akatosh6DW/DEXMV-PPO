"""Video selection must retain fixed failures instead of showing only wins."""

import unittest

from orca_formal_protocol import SEEDS
from render_orca_formal import select_cases


class FormalVideoTests(unittest.TestCase):
    def test_fixed_cases_are_preserved_and_failure_selection_is_explicit(self):
        reports = {f"s{seed}_{method}_{tactile}": dict(episodes=dict(success30=[True] * 64))
                   for seed in SEEDS for method in ("bc", "ppo", "guided", "residual") for tactile in ("off", "on")}
        reports["s301_residual_off"]["episodes"]["success30"][0] = False
        reports["s302_ppo_off"]["episodes"]["success30"][3] = False
        cases = select_cases(reports)
        fixed = [c for c in cases if c["reason"] == "fixed_seed301_reset25000"]
        self.assertEqual(len(fixed), 8)
        self.assertTrue(any(c["run"] == "s301_residual_off" and c["episode"] == 0 for c in cases))
        self.assertTrue(any(c["run"] == "s302_ppo_off" and c["episode"] == 3
                            and "failure" in c["reason"] for c in cases))
        self.assertTrue(any(c["run"] == "s301_residual_off" and c["episode"] == 1
                            and "success_selected" in c["reason"] for c in cases))


if __name__ == "__main__":
    unittest.main()
