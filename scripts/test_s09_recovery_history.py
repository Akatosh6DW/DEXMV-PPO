import unittest

import numpy as np

from run_s09_bc_without_action_history import mask_arrays


class RecoveryHistoryTests(unittest.TestCase):
    def test_physical_feedback_retained(self):
        arrays = tuple(np.ones((2, 123 if i%2 == 0 else 17)) for i in range(6))
        result = mask_arrays(arrays, False)
        for index in range(6):
            actual = np.asarray(result[index])
            if index%2:
                np.testing.assert_array_equal(actual, arrays[index])
            else:
                np.testing.assert_array_equal(actual[:,34:68], 0)
                np.testing.assert_array_equal(actual[:,:34], 1)
                np.testing.assert_array_equal(actual[:,68:], 1)
        with self.assertRaises(ValueError):
            mask_arrays(arrays, True)


if __name__ == "__main__":
    unittest.main()
