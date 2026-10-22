import unittest
import numpy as np
from collect_s09_recovery_demo import merge_training


class RecoveryCollectionTests(unittest.TestCase):
    def test_validation_is_not_moved_into_training(self):
        original = {k:np.arange(8)[:,None] for k in ("obs", "action", "active")}
        extra = {k:np.array([[99],[100]]) for k in original}
        result = merge_training(original, extra, 6)
        for key in result:
            np.testing.assert_array_equal(result[key].ravel(), [0,1,2,3,4,5,99,100,6,7])
            np.testing.assert_array_equal(original[key].ravel(), np.arange(8))
        with self.assertRaises(ValueError):
            merge_training(original, extra, 8)


if __name__ == "__main__":
    unittest.main()
