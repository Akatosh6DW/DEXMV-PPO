"""Verify measured-pose conditioning cannot change actions or physical dynamics."""

from pathlib import Path
import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_bc_conditioned import ModeConditionedGrasp
from orca_s09_env import OrcaS09Grasp


class ConditioningTests(unittest.TestCase):
    def test_mode_is_remembered_but_not_added_to_action(self):
        model = Path(__file__).resolve().parents[1]/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
        baseline, changed = OrcaS09Grasp(model), ModeConditionedGrasp(model)
        a, b = baseline.reset(jax.random.PRNGKey(111)), changed.reset(jax.random.PRNGKey(111))
        mode = np.asarray(b.obs["state"][95:101])
        self.assertEqual(float(mode.sum()), 1.)
        np.testing.assert_array_equal(changed.reference_ctrl, 0.)
        for _ in range(4):
            a, b = baseline.step(a, jp.zeros(17)), changed.step(b, jp.zeros(17))
            np.testing.assert_array_equal(a.data.integration_words, b.data.integration_words)
            np.testing.assert_array_equal(b.obs["state"][95:101], mode)
            np.testing.assert_array_equal(b.info["requested_residual"], 0.)
            np.testing.assert_array_equal(a.info["desired_ctrl"], b.info["desired_ctrl"])


if __name__ == "__main__":
    unittest.main()
