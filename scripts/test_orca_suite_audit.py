"""Independent trajectory audit catches tampered outcomes and residuals."""

import json
from pathlib import Path
import unittest

import numpy as np

from audit_orca_suite import check_preview

ROOT = Path(__file__).resolve().parents[1]


class AuditTests(unittest.TestCase):
    def setUp(self):
        run = ROOT / "artifacts/orca_benchmark_suite/smoke_v2"
        step = run / "step_000000000"
        self.config = json.loads((run / "run_config.json").read_text())
        self.episodes = json.loads((step / "episodes.json").read_text())
        with np.load(step / "preview_trajectories.npz") as z:
            self.trace = {k: z[k] for k in z.files}

    def test_actual_smoke(self):
        check_preview(self.trace, self.episodes, self.config)

    def test_incorrect_outcome_rejected(self):
        self.episodes[0]["success30"] = not self.episodes[0]["success30"]
        with self.assertRaises(AssertionError):
            check_preview(self.trace, self.episodes, self.config)

    def test_false_clean_success_rejected(self):
        self.trace["substep_penetration"][0, 300] = .004
        with self.assertRaises(AssertionError):
            check_preview(self.trace, self.episodes, self.config)

    def test_nonzero_zero_controller_rejected(self):
        self.trace["requested_residual"][0, 20, 0] = .04
        with self.assertRaises(AssertionError):
            check_preview(self.trace, self.episodes, self.config)


if __name__ == "__main__":
    unittest.main()
