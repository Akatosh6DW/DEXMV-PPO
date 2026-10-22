"""Parameter verification must detect changes rather than trust metadata."""

import unittest
import numpy as np
from verify_s09_learned_residual_warmup import difference


class ParameterDifferenceTests(unittest.TestCase):
    def test_equal_and_changed(self):
        a = dict(w=np.array([1., 2.], np.float32))
        b = dict(w=np.array([1., 3.], np.float32))
        self.assertEqual(difference(a, a)["changed_elements"], 0)
        self.assertEqual(difference(a, b)["changed_elements"], 1)
        self.assertEqual(difference(a, b)["maximum_absolute_difference"], 1.)

    def test_invalid_tree(self):
        with self.assertRaises(ValueError):
            difference(dict(w=np.ones(2)), dict(v=np.ones(2)))
        with self.assertRaises(ValueError):
            difference(dict(w=np.ones(2)), dict(w=np.ones(3)))
        with self.assertRaises(ValueError):
            difference(dict(w=np.ones(1)), dict(w=np.array([np.nan])))


if __name__ == "__main__":
    unittest.main()
