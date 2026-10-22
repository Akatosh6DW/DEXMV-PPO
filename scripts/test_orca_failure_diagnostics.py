"""A retained-but-low cube must not be labelled a drop or a success."""

import unittest
import numpy as np

from orca_failure_diagnostics import classify


class FailureDiagnosticsTests(unittest.TestCase):
    def test_retained_height_failure_and_invalid_reset(self):
        shape = (3, 800)
        trace = dict(alive=np.ones(shape, bool), hold=np.ones(shape) * 35,
                     released=np.ones(shape, bool), thumb=np.ones(shape), contacts=np.ones(shape) * 2,
                     forbidden=np.zeros(shape), cube_z=np.ones(shape), spawn_z=np.ones(shape),
                     initial_penetration_m=np.zeros(shape))
        trace["cube_z"][1, -1] -= .015
        trace["hold"][1, -1] = 0
        trace["initial_penetration_m"][2] = .004
        records = classify(trace)["episodes"]
        self.assertEqual([r["reason"] for r in records], ["success", "retained_but_height_invalid", "invalid_initial_geometry"])
        self.assertTrue(records[1]["alive_at_end"])
        self.assertIsNone(records[1]["termination_time_s"])


if __name__ == "__main__":
    unittest.main()
