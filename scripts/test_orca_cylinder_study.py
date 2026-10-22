"""Cylinder experiment boundaries and reference calibration invariants."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import numpy as np
from refine_orca_cylinder_reference import ramp
from run_orca_cylinder_job import validate


class ProtocolTests(unittest.TestCase):
    def job(self, **changes):
        return dict(dict(method="ppo", steps=655360, evals=5, eval_count=16, eval_start=100000), **changes)

    def test_valid_budget(self):
        validate(self.job())

    def test_no_final_test_seeds_during_training(self):
        with self.assertRaises(ValueError):
            validate(self.job(eval_start=110000))

    def test_no_selected_old_checkpoint_training(self):
        with self.assertRaises(ValueError):
            validate(self.job(checkpoint="old.pkl"))

    def test_no_training_under_held_out_condition(self):
        with self.assertRaises(ValueError):
            validate(self.job(benchmark="B1_position"))

    def test_bc_requires_bound_data(self):
        with self.assertRaises(ValueError):
            validate(self.job(method="guided"))

    def test_reference_acquisition_is_unchanged(self):
        np.testing.assert_array_equal(ramp([0., 1., 1.5]), 0.)
        np.testing.assert_allclose(ramp([2.25, 3., 40.]), [.5, 1., 1.])


if __name__ == "__main__":
    unittest.main()
