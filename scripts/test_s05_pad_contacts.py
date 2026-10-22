"""Reject side/proximal touches even if they belong to a fingertip body."""

import unittest

from s05_pad_contacts import is_distal_pad


class PadTests(unittest.TestCase):
    def test_distal_front(self):
        self.assertTrue(is_distal_pad("orca_index_tip_proxy_fit_03", [0, 0, .038], [-1, 0, 0], 1))

    def test_lateral_tip_rejected(self):
        self.assertFalse(is_distal_pad("orca_index_tip_proxy_fit_03", [0, 0, .038], [-.26, -.94, .22], 1))

    def test_proximal_rejected(self):
        self.assertFalse(is_distal_pad("orca_index_prox_proxy_fit_00", [0, 0, .038], [-1, 0, 0], 1))
        self.assertFalse(is_distal_pad("orca_index_tip_proxy_fit_01", [0, 0, .012], [-1, 0, 0], 1))

    def test_thumb_front_only(self):
        self.assertTrue(is_distal_pad("orca_thumb_tip_proxy_fit_02", [0, 0, .024], [-1, 0, 0], 0))
        self.assertFalse(is_distal_pad("orca_thumb_tip_proxy_fit_02", [0, 0, .024], [1, 0, 0], 0))


if __name__ == "__main__":
    unittest.main()
