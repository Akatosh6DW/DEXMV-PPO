"""No-physics tests of the complete preregistered S03 job matrix."""

import unittest

from run_orca_s03_confirmation import BASE, training_jobs, evaluation_jobs, select_force, FORCES


class MatrixTests(unittest.TestCase):
    def test_training_budget_and_population(self):
        jobs = list(training_jobs(BASE))
        self.assertEqual(len(jobs), 40)
        self.assertEqual(len({j["output"] for j in jobs}), 40)
        self.assertEqual(sum(j["steps"] for j in jobs), 26214400)
        self.assertTrue(all(j["eval_start"] == 61000 for j in jobs))
        self.assertTrue(all(j["evals"] == 9 and j["steps"] == 655360 for j in jobs))
        self.assertTrue(all("checkpoint" not in j for j in jobs))

    def test_full_matrix_and_checkpoint_paths(self):
        jobs = list(evaluation_jobs(BASE, .6))
        training = {j["output"] for j in training_jobs(BASE)}
        self.assertEqual(len(jobs), 738)
        self.assertEqual(len({j["output"] for j in jobs}), 738)
        self.assertEqual(sum(4 if j.get("paired_replay") else 1 for j in jobs)*200, 153600)
        for j in jobs:
            self.assertIn(j["eval_start"], (71000, 72000, 73000))
            self.assertEqual(j["eval_count"], 200)
            if j["method"] == "zero":
                self.assertNotIn("checkpoint", j)
                continue
            path = j["checkpoint"]
            directory = path.rsplit("/", 2)[0] if "/step_" in path else path.rsplit("/", 1)[0]
            self.assertIn(directory, training)
        self.assertEqual(sum(j["method"] == "zero" for j in jobs), 8)
        self.assertEqual(sum(j.get("shape") is not None for j in jobs), 182)
        self.assertEqual(sum(bool(j.get("paired_replay")) for j in jobs), 10)

    def test_force_choice_uses_reference_survival_only_and_ties_low(self):
        def rows(n):
            return [dict(valid_before_push=True, valid_one_second_after_push=i < n) for i in range(4)]
        result = select_force({.15: rows(4), .3: rows(3), .6: rows(1), 1.2: rows(0)})
        self.assertEqual(result["selected_force_n"], .3)
        self.assertTrue(result["informative"])
        same = select_force({f: rows(4) for f in FORCES})
        self.assertEqual(same["selected_force_n"], .15)
        self.assertFalse(same["informative"])
        with self.assertRaises(ValueError):
            select_force({f: [dict(valid_before_push=False)] for f in FORCES})


if __name__ == "__main__":
    unittest.main()
