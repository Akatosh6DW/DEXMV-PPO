import unittest

import numpy as np

from analyze_s09_angular_feedback import paired_initials


class PairedInitialTests(unittest.TestCase):
    def test_same_initials_pass(self):
        a = dict(seeds=np.arange(3), qpos=np.zeros((3, 24)))
        paired_initials(a, {k: v.copy() for k, v in a.items()})

    def test_seed_or_state_change_fails(self):
        a = dict(seeds=np.arange(3), qpos=np.zeros((3, 24)))
        for key in a:
            b = {k: v.copy() for k, v in a.items()}
            b[key].flat[0] += 1
            with self.assertRaises(AssertionError):
                paired_initials(a, b)


if __name__ == "__main__":
    unittest.main()
