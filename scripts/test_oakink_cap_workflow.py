"""Job coverage and honest cap video selection."""

import unittest

from oakink_cap_protocol import SEEDS, recipes
from render_oakink_cap_formal import select_cases
from run_oakink_cap_evaluations import jobs


class CapWorkflowTests(unittest.TestCase):
    def test_every_method_seed_and_intervention_is_present(self):
        training = recipes()
        self.assertEqual(len(training), 20)
        self.assertEqual(len({(r["seed"], r["method"]) for r in training}), 20)
        evaluations = jobs()
        self.assertEqual(len(evaluations), 110)
        self.assertEqual(len({j["label"] for j in evaluations}), 110)
        self.assertEqual(sum(j["baseline"] is not None for j in evaluations), 10)

    def test_all_failure_videos_do_not_invent_success_case(self):
        reports = {f"s{s}_{m}": dict(independent_audit=dict(episodes=dict(success=[False] * 64)))
                   for s in SEEDS for m in ("bc", "pure", "guided", "residual")}
        cases = select_cases(reports)
        self.assertEqual(len(cases), 4)
        self.assertTrue(all(c["episode"] == 0 and c["run"].startswith("s401_") for c in cases))
        reports["s403_residual"]["independent_audit"]["episodes"]["success"][2] = True
        cases = select_cases(reports)
        self.assertEqual(len(cases), 5)
        self.assertEqual(cases[-1]["run"], "s403_residual")
        self.assertEqual(cases[-1]["episode"], 2)
        self.assertIn("not_rate_estimate", cases[-1]["reason"])


if __name__ == "__main__":
    unittest.main()
