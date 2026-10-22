"""Training randomization must not silently change the matched evaluation."""

import unittest

import jax
import numpy as np

from run_s09_training_jitter import training_environment
from orca_s09_env import OrcaS09Grasp
from run_s09_bc_conditioned import ModeConditionedGrasp
from s09_suspension import OUT


class TrainingJitterTests(unittest.TestCase):
    def test_only_reset_position_range_changes_for_both_architectures(self):
        for cls, arm in [(ModeConditionedGrasp, "pure"), (OrcaS09Grasp, "residual")]:
            ref = OUT/"inputs/reference.npz" if arm == "residual" else None
            evaluation = cls(OUT/"inputs/model.xml", arm=arm, reference=ref)
            training = training_environment(evaluation, .005)
            self.assertIsNot(training.native, evaluation.native)
            self.assertEqual(evaluation.jitter, .002)
            self.assertEqual(training.jitter, .005)
            cfg = training._config.to_dict()
            cfg["jitter"] = .002
            self.assertEqual(cfg, evaluation._config.to_dict())
            for seed in (3, 7, 11):
                a = evaluation.reset(jax.random.PRNGKey(seed))
                b = training.reset(jax.random.PRNGKey(seed))
                qa, qb = np.asarray(a.data.qpos), np.asarray(b.data.qpos)
                oid = evaluation.oid
                keep = np.ones(len(qa), dtype=bool)
                keep[oid:oid+3] = False
                np.testing.assert_array_equal(qa[keep], qb[keep])
                np.testing.assert_array_equal(a.data.qvel, b.data.qvel)
                self.assertEqual(int(a.info["reference_index"]), int(b.info["reference_index"]))
                self.assertGreater(float(np.max(np.abs(qa-qb))), 0.)
                self.assertLessEqual(float(np.max(np.abs(qa-qb))), .003001)

    def test_invalid_jitter_rejected(self):
        for value in (0, .001, .02, float("nan")):
            with self.assertRaises(ValueError):
                training_environment(None, value)


if __name__ == "__main__":
    unittest.main()
