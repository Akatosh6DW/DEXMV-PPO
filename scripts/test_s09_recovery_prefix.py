import unittest

import numpy as np

from filter_s09_recovery_prefix import filter_prefix


class RecoveryPrefixTests(unittest.TestCase):
    def test_only_added_prefix_mask_changes(self):
        data = dict(obs=np.ones((6, 8, 123)), action=np.ones((6, 8, 17)), active=np.ones((6, 8), bool))
        filtered = filter_prefix(data, 4, 1, 3)
        expected = np.ones((6, 8), bool)
        expected[3, :3] = False
        np.testing.assert_array_equal(filtered["active"], expected)
        self.assertTrue(data["active"].all())
        for key in ("obs", "action"):
            np.testing.assert_array_equal(filtered[key], data[key])

    def test_bad_split_rejected(self):
        data = dict(obs=np.ones((6, 8, 123)), active=np.ones((6, 8), bool))
        for args in ((4, 4, 3), (6, 1, 3), (4, 1, 8), (4, 0, 3)):
            with self.assertRaises(ValueError):
                filter_prefix(data, *args)


if __name__ == "__main__":
    unittest.main()
