import unittest

import jax.numpy as jp
import numpy as np

from run_s09_residual_phase_probe import gate_logits


class PhaseProbeTests(unittest.TestCase):
    def test_exact_release_boundary(self):
        obs = dict(state=jp.zeros((3,123)).at[:,112].set(jp.array([59,60,61])/800))
        logits = jp.ones((3,34))
        before = np.asarray(gate_logits(logits,obs,"before_release"))
        after = np.asarray(gate_logits(logits,obs,"after_release"))
        np.testing.assert_array_equal(before[:,:17],np.broadcast_to([[1],[0],[0]],(3,17)))
        np.testing.assert_array_equal(after[:,:17],1-before[:,:17])
        np.testing.assert_array_equal(before[:,17:],1)
        np.testing.assert_array_equal(after[:,17:],1)
        with self.assertRaises(ValueError):
            gate_logits(logits,obs,"automatic")


if __name__ == "__main__":
    unittest.main()
