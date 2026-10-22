import unittest

import numpy as np

from diagnose_s09_critic_gradient_conflict import cosine_matrix


class GradientConflictTests(unittest.TestCase):
    def test_direction_scale_and_zero(self):
        result = cosine_matrix([[1,0],[-2,0],[0,1],[0,0]])
        np.testing.assert_array_equal(result[:3,:3],[[1,-1,0],[-1,1,0],[0,0,1]])
        self.assertTrue(np.isnan(result[3]).all())
        self.assertTrue(np.isnan(result[:,3]).all())

    def test_invalid(self):
        for values in ([1,2],[[1,np.nan]],[[1,np.inf]]):
            with self.assertRaises(ValueError):
                cosine_matrix(values)


if __name__ == "__main__":
    unittest.main()
