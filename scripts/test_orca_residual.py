"""Small, CPU-only tests for reference and scoring contracts."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from orca_residual_controls import compose, load_reference
from orca_residual_metrics import summarize


class ResidualTests(unittest.TestCase):
    def test_no_accumulation_and_clipping(self):
        base = np.array([.5, -.5])
        for _ in range(800):
            target, correction = compose(base, np.array([20., -20.]), .08, -1, 1)
            np.testing.assert_allclose(target, [.58, -.58])
            np.testing.assert_allclose(correction, [.08, -.08])
        target, _ = compose(np.array([.99]), np.ones(1), .08, -1, 1)
        np.testing.assert_allclose(target, [1])

    def test_reference_slew_and_residual_tube(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"r.npz"
            np.savez(path, ctrl=[[0., 0.], [.9, -.9]], time=[0., .05])
            base, raw = load_reference(path, [-1, -1], [1, 1], [0, 0], .05, 800, .1)
            self.assertEqual(raw[0, 0], np.float32(.9))
            np.testing.assert_allclose(base[0], [.1, -.1])
            actual = np.zeros(2)
            rng = np.random.default_rng(12)
            for ref in base:
                desired, _ = compose(ref, rng.uniform(-3, 3, 2), .08, -1, 1)
                actual = np.clip(desired, actual-.1, actual+.1)
                self.assertLessEqual(np.max(np.abs(actual-ref)), .080001)
            np.savez(path, ctrl=[[0., 0.], [.9, -.9]], time=[0., 0.])
            with self.assertRaises(ValueError):
                load_reference(path, [-1, -1], [1, 1], [0, 0], .05, 800, .1)
            np.savez(path, ctrl=[[0., 0.], [2., -.9]], time=[0., .05])
            with self.assertRaises(ValueError):
                load_reference(path, [-1, -1], [1, 1], [0, 0], .05, 800, .1)

    def test_independent_continuous_score_and_late_drop(self):
        n = 800
        trace = dict(alive=np.ones((3, n), bool), thumb=np.ones((3, n)),
                     contacts=np.full((3, n), 2), forbidden=np.zeros((3, n)),
                     released=np.ones((3, n)), cube_z=np.ones((3, n)),
                     spawn_z=np.ones((3, n)), reward=np.ones((3, n)),
                     ctrl=np.zeros((3, n, 2)))
        trace["forbidden"][1, 400] = 1
        trace["alive"][2, 700:] = False
        hold = np.zeros((3, n))
        for i in range(n):
            valid = trace["alive"][:, i] & (trace["forbidden"][:, i] == 0)
            hold[:, i] = np.where(valid, (hold[:, i-1] if i else 0)+.05, 0)
        trace["hold"] = hold
        row, holds, success = summarize(trace, "pure", .08, [-1, -1], [1, 1])
        np.testing.assert_allclose(holds, [40, 20, 35])
        np.testing.assert_array_equal(success, [True, False, False])
        self.assertEqual(row["success30_any"], 2/3)
        trace["hold"][1, -1] = 40
        with self.assertRaises(AssertionError):
            summarize(trace, "pure", .08, [-1, -1], [1, 1])

    def test_fabricated_residual_is_rejected(self):
        shape = (1, 800)
        trace = {k: np.ones(shape) for k in ("thumb", "released", "cube_z", "spawn_z", "reward")}
        trace.update(alive=np.ones(shape, bool), contacts=np.full(shape, 2),
                     forbidden=np.zeros(shape), hold=np.arange(1, 801)[None, :]*.05)
        for k in ("ctrl", "action", "video_base", "requested_residual", "effective_residual", "desired_ctrl"):
            trace[k] = np.zeros((1, 800, 2))
        summarize(trace, "residual", .08, [-1, -1], [1, 1])
        trace["ctrl"][:, :, 0] = .09
        trace["effective_residual"][:, :, 0] = .09
        with self.assertRaises(AssertionError):
            summarize(trace, "residual", .08, [-1, -1], [1, 1])


if __name__ == "__main__":
    unittest.main()
