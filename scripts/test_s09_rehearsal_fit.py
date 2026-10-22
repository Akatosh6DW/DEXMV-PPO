"""The full-dataset diagnostic must weight uneven batches by frame count."""

import unittest
import numpy as np
from measure_s09_rehearsal_fit import mean_error


class RehearsalFitTests(unittest.TestCase):
    def test_batches_do_not_reweight_the_last_frames(self):
        obs = np.arange(1, 6, dtype=np.float32)[:, None]
        actions = np.zeros((5, 17), np.float32)
        apply = lambda stats, policy, values: np.repeat(values, 34, axis=1)
        for size in (1, 2, 8):
            self.assertAlmostEqual(mean_error(apply, None, None, obs, actions, size), 11.)

    def test_nonfinite_outputs_and_targets_rejected(self):
        obs = np.zeros((1, 123), np.float32)
        actions = np.zeros((1, 17), np.float32)
        for bad_output in (True, False):
            apply = lambda *args: np.full((1, 34), np.nan if bad_output else 0)
            target = actions.copy()
            if not bad_output:
                target[0, 0] = np.nan
            with self.assertRaises(ValueError):
                mean_error(apply, None, None, obs, target)


if __name__ == "__main__":
    unittest.main()
