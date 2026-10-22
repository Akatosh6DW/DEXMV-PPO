import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_recovery_window_bc import sample_late, window_indices


class RecoveryWindowTests(unittest.TestCase):
    def test_only_new_train_active_window(self):
        active = np.ones((5, 100), bool)
        active[2, 40] = False
        indices, size = window_indices(active, 2, 4)
        late = active & (np.arange(5)[:, None] < 4) & (np.arange(100)[None, :] >= 30)
        pairs = np.argwhere(late)[indices]
        self.assertEqual(size, 279)
        self.assertEqual(len(indices), 79)
        self.assertTrue(np.all((pairs[:, 0] >= 2) & (pairs[:, 0] < 4)))
        self.assertTrue(np.all((pairs[:, 1] >= 40) & (pairs[:, 1] < 80)))

    def test_control_and_unchanged_tail(self):
        key = jax.random.PRNGKey(9)
        eligible = jp.array([2, 7, 11])
        control = np.asarray(jax.random.randint(key, (128,), 0, 100))
        np.testing.assert_array_equal(sample_late(key, 100, eligible, 0), control)
        changed = np.asarray(sample_late(key, 100, eligible, 16))
        np.testing.assert_array_equal(changed[16:], control[16:])
        self.assertTrue(np.isin(changed[:16], eligible).all())

    def test_empty_window_rejected(self):
        with self.assertRaises(ValueError):
            window_indices(np.zeros((5, 100), bool), 2, 4)


if __name__ == "__main__":
    unittest.main()
