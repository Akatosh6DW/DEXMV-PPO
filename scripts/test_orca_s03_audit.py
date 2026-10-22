"""Audit rejection tests using S03 smoke artifacts, not final test data."""

import json
from pathlib import Path
import unittest

import numpy as np

from audit_orca_s03 import verify_trace
from render_orca_s03 import selections
from run_orca_s03_confirmation import BASE, evaluation_jobs

ROOT = Path(__file__).resolve().parents[1]


class AuditTests(unittest.TestCase):
    def setUp(self):
        run = ROOT / "artifacts/orca_benchmark_suite/S03/smoke_replay_v1"
        self.config = json.loads((run / "run_config.json").read_text())
        self.episodes = json.loads((run / "pushed_closed/episodes.json").read_text())
        with np.load(run / "pushed_closed/preview_trajectories.npz") as z:
            self.trace = dict(z)

    def test_real_smoke_trace(self):
        verify_trace(self.trace, self.episodes, self.config, .6)

    def test_wrong_force_rejected(self):
        with self.assertRaises(AssertionError):
            verify_trace(self.trace, self.episodes, self.config, 0.)

    def test_wrong_clean_outcome_rejected(self):
        self.episodes[0]["clean_success30"] = not self.episodes[0]["clean_success30"]
        with self.assertRaises(AssertionError):
            verify_trace(self.trace, self.episodes, self.config, .6)

    def test_wrong_residual_rejected(self):
        self.trace["requested_residual"][0, 20, 0] += .04
        with self.assertRaises(AssertionError):
            verify_trace(self.trace, self.episodes, self.config, .6)

    def test_video_selection_is_outcome_blind(self):
        chosen = list(selections(list(evaluation_jobs(BASE, .6))))
        self.assertEqual(len(chosen), 50)
        self.assertTrue(all(j["seed"] == 21001 and j["budget"] in (0, 655360) for j, _ in chosen))
        self.assertEqual(sum(m in ("pushed_closed", "pushed_replay") for _, m in chosen), 2)


if __name__ == "__main__":
    unittest.main()
