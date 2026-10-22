"""Check report aggregation independently of simulation and model selection."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from summarize_orca_residual import read_evaluation


class ReportTests(unittest.TestCase):
    def test_best_hold_is_not_final_uninterrupted_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            checkpoint = run / "step_000000000"
            checkpoint.mkdir()
            (run / "run_config.json").write_text(json.dumps({"bound": .08}))
            # Episode 0 had a long hold, briefly lost contact, then recovered.
            # Episode 1 held continuously; episode 2 eventually dropped.
            hold = np.zeros((3, 800))
            hold[0, :660] = np.arange(1, 661) * .05
            hold[0, 700:] = np.arange(1, 101) * .05
            hold[1] = np.arange(1, 801) * .05
            hold[2, :680] = np.arange(1, 681) * .05
            alive = np.ones((3, 800), dtype=bool)
            alive[2, 680:] = False
            contact = np.full((3, 800), 2)
            contact[0, 660:700] = 0
            contact[2, 680:] = 0
            np.savez(checkpoint / "trajectories.npz", hold=hold, alive=alive,
                     thumb=(contact > 0).astype(int), contacts=contact,
                     forbidden=np.zeros((3, 800)), released=np.ones((3, 800)),
                     cube_z=np.ones((3, 800)), spawn_z=np.ones((3, 800)),
                     requested_residual=np.zeros((3, 800, 17)),
                     effective_residual=np.zeros((3, 800, 17)))
            (checkpoint / "episodes.json").write_text(json.dumps({
                "seeds": [2000, 2001, 2002], "hold_sec": [33., 40., 34.],
                "success30": [True, True, False],
            }))
            result = read_evaluation(run)
            self.assertEqual(result["episode_count"], 3)
            self.assertEqual(result["success_count"], 2)
            self.assertAlmostEqual(result["success"], 2 / 3)
            self.assertAlmostEqual(result["final_continuous30"], 1 / 3)
            self.assertAlmostEqual(result["mean_final_continuous_hold"], 15.)
            self.assertAlmostEqual(result["mean_best_hold"], 107 / 3)
            self.assertAlmostEqual(result["diagnostics"]["dropped_fraction"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
