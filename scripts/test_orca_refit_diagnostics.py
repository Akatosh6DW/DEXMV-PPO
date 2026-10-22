"""Failure analysis must exclude post-terminal simulation tails."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from diagnose_orca_refit import diagnose


class DiagnosticTests(unittest.TestCase):
    def test_early_failure_has_no_fake_release_displacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            shape = (1, 80)
            active = np.zeros(shape, bool)
            active[:, :10] = True
            qpos = np.zeros((*shape, 24))
            qpos[:, :, -4] = 1
            contacts = np.zeros(shape)
            contacts[:, -1] = 5
            np.savez_compressed(path, active=active, alive=active, qpos=qpos,
                                qvel=np.zeros((*shape, 23)), released=np.zeros(shape),
                                contacts=contacts, thumb=np.zeros(shape), forbidden=np.zeros(shape),
                                cube_z=np.zeros(shape), spawn_z=np.ones(shape), hold=np.zeros(shape))
            result = diagnose(path)
            record = result["episodes_detail"][0]
            self.assertIsNone(record["displacement_since_release_mm"])
            self.assertIsNone(record["best_hold_end_rotation_deg"])
            self.assertEqual(record["last_active_contacts"], 0)
            self.assertEqual(record["last_active_step"], 9)


if __name__ == "__main__":
    unittest.main()
