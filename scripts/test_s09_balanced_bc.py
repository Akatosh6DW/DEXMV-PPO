"""Mode balancing must preserve labels and keep validation out of sampling."""

import unittest

import numpy as np

from run_s09_balanced_bc import balance_arrays, balance_stratum


class BalancedBCTests(unittest.TestCase):
    def examples(self):
        modes = np.array([0, 0, 0, 0, 2, 5])
        obs = np.zeros((len(modes), 123), np.float32)
        obs[:, 95:101] = np.eye(6)[modes]
        obs[:, 0] = np.arange(len(modes))
        actions = np.broadcast_to(np.arange(len(modes))[:, None], (len(modes), 17)).copy()
        return obs, actions

    def test_exact_balancing_and_pairing(self):
        obs, actions = self.examples()
        before = obs.copy()
        x, y, meta = balance_stratum(obs, actions)
        np.testing.assert_array_equal(np.bincount(np.asarray(x[:, 95:101]).argmax(1), minlength=6), [4, 0, 4, 0, 0, 4])
        np.testing.assert_array_equal(x[:, 0], y[:, 0])
        np.testing.assert_array_equal(obs, before)
        self.assertEqual(meta["absent_modes"], [1, 3, 4])

    def test_validation_identity(self):
        obs, actions = self.examples()
        val_obs, val_action = np.full((2, 123), 999), np.full((2, 17), 999)
        result, _ = balance_arrays((obs, actions, obs, actions, val_obs, val_action))
        self.assertIs(result[4], val_obs)
        self.assertIs(result[5], val_action)
        self.assertLess(np.max(result[0]), 999)

    def test_bad_modes_and_replication_limit(self):
        obs, actions = self.examples()
        with self.assertRaises(ValueError):
            balance_stratum(obs, actions, max_rows=5)
        obs[0, 95:101] = .5
        with self.assertRaises(ValueError):
            balance_stratum(obs, actions)


if __name__ == "__main__":
    unittest.main()
