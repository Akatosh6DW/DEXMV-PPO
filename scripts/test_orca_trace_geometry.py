"""Active-frame and distance semantics for replay geometry checks."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_orca_trace_geometry import audit


class TraceGeometryTests(unittest.TestCase):
    def test_inactive_deep_contact_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.xml"
            model.write_text('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
                             '<body><freejoint/><geom name="cube" type="box" size=".01 .01 .01"/>'
                             '</body></worldbody></mujoco>')
            trace = root / "trace.npz"
            qpos = np.array([[[0, 0, .009, 1, 0, 0, 0], [0, 0, -.04, 1, 0, 0, 0]]])
            np.savez(trace, active=[[True, False]], qpos=qpos, qvel=np.zeros((1, 2, 6)),
                     ctrl=np.zeros((1, 2, 0)), mocap=np.zeros((1, 2, 0, 3)))
            result = audit(model, trace)
            self.assertEqual(result["states"], 1)
            self.assertAlmostEqual(result["max_penetration_m"], .001)
            self.assertEqual(result["fraction_over_3mm"], 0.)
            self.assertEqual(result["peak"]["geom"], "floor")


if __name__ == "__main__":
    unittest.main()
