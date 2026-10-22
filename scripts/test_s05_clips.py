"""S05 failure cuts include loss of an established distal-pad grasp."""

import unittest
import numpy as np

from render_s05_clips import s05_clip_endpoint
from test_presentation_clip_endpoints import trace


class S05ClipTests(unittest.TestCase):
    def test_height_or_pad_loss_stops_failed_clip(self):
        t = trace()
        t["hold"][:] = np.arange(1, 201) * .05
        t["hold"][150:] = 0
        self.assertEqual(s05_clip_endpoint(t, False)[0], 150)

    def test_earlier_audit_failure_has_priority(self):
        t = trace()
        t["hold"][:] = np.arange(1, 201) * .05
        t["hold"][150:] = 0
        t["substep_forbidden"][120] = 1
        self.assertEqual(s05_clip_endpoint(t, False)[0], 120)

    def test_success_keeps_four_second_cut(self):
        t = trace()
        t["released"][:59] = 0
        self.assertEqual(s05_clip_endpoint(t, True)[0], 139)


if __name__ == "__main__":
    unittest.main()
