"""Query ordering must not use teacher outcomes or validation trajectories."""

import unittest
from pathlib import Path
import tempfile
import numpy as np

from collect_s09_student_corrections import select_queries, canonical_plan, runner
from run_s09_bc_conditioned import ModeConditionedGrasp


class StudentCorrectionTests(unittest.TestCase):
    def test_order_limit_and_prefix_alive(self):
        episodes = [dict(success30=False) for _ in range(10)]
        episodes[0]["success30"] = True
        trace = dict(active=np.ones((10, 800), bool), alive=np.ones((10, 800), bool))
        trace["alive"][1, 39] = False
        trace["active"][2, 10] = False
        self.assertEqual(select_queries(episodes, trace), [3, 4, 5, 6, 7, 8])
        self.assertEqual(select_queries(episodes, trace, limit=1), [3])

    def test_no_eligible_cases_is_explicitly_empty(self):
        episodes = [dict(success30=True)]
        trace = dict(active=np.ones((1, 800), bool), alive=np.ones((1, 800), bool))
        self.assertEqual(select_queries(episodes, trace), [])

    def test_canonical_collection_replays_verified_training_plan(self):
        root = Path(__file__).resolve().parents[1]
        model = root/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
        path = root/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018/D36_early_recovery_dataset/seed90523_canonical_trace.npz"
        with np.load(path, allow_pickle=False) as z:
            targets = z["desired_ctrl"][0]
        env = ModeConditionedGrasp(model, "pure")
        net, _ = runner.initialize_matched(env, 13112, .03)
        before = runner.networks.make_inference_fn
        with tempfile.TemporaryDirectory() as directory:
            report, trace = canonical_plan(env, net, 90523, targets, Path(directory), model)
            self.assertTrue(report["qualified"])
            self.assertEqual(trace["obs"].shape, (1, 800, 123))
        self.assertIs(runner.networks.make_inference_fn, before)


if __name__ == "__main__":
    unittest.main()
