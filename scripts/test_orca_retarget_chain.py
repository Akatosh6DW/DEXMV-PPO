"""Distinguish upstream, downstream and total object-frame marker errors."""

from pathlib import Path
import unittest

import numpy as np

from audit_orca_retarget_chain import audit, summarize


class RetargetChainTests(unittest.TestCase):
    def test_norm_summary_reports_per_finger_not_averaged_hand(self):
        delta = np.array([[[.003, .004, 0], [0, 0, .010]],
                          [[0, 0, 0], [0, 0, .020]]])
        result = summarize(delta)
        np.testing.assert_allclose(result["mean_mm"], [2.5, 15])
        np.testing.assert_allclose(result["endpoint_mm"], [0, 20])

    def test_downstream_small_error_does_not_imply_total_small_error(self):
        root = Path(__file__).resolve().parents[1] / "artifacts/orca_collision_refit"
        model = root / "O20_hic53_elliptic/cube.xml"
        if not model.exists():
            self.skipTest("Matched reference not present")
        report, arrays = audit(model, model.parent / "reference.npz",
                               root / "O01_hic53_source_object/source_correspondence.npz")
        self.assertEqual(report["source_frames"], [30, 95])
        self.assertLess(report["stages"]["adroit_to_orca"]["endpoint_mm"][1], 1)
        self.assertGreater(report["stages"]["human_to_orca_total"]["endpoint_mm"][1], 7)
        self.assertEqual(arrays["human"].shape, (66, 3, 3))


if __name__ == "__main__":
    unittest.main()
