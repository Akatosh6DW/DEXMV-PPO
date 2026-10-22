import unittest

import numpy as np

from oakink_cap_failure_diagnostics import classify


class CapFailureDiagnosticsTest(unittest.TestCase):
    def test_released_is_not_success_and_post_termination_is_ignored(self):
        keys = ("alive", "active", "released", "forbidden", "hand_penetration", "reset_penetration",
                "paired", "lift", "bottle_contact", "floor_contact", "success", "force", "hold_sec")
        t = {key: np.zeros((8, 400)) for key in keys}
        t["alive"][:] = t["active"][:] = 1
        t["released"][1:] = 1
        t["paired"][1:] = 1
        t["lift"][1:] = .02
        t["success"][1, -1] = 1
        t["hold_sec"][1, -1] = 1
        t["forbidden"][2, 3] = 1
        t["hand_penetration"][3, 3] = .006
        t["alive"][4, 50:] = 0
        t["active"][4, 51:] = 0
        t["floor_contact"][4, 50] = 1
        t["forbidden"][4, 51:] = 1
        t["success"][4, 51:] = 1
        t["force"][4, 51:] = 1e8
        t["paired"][5, -1] = 0
        t["lift"][6, -1] = .01
        rows = classify(t)["episodes"]
        self.assertEqual([r["reason"] for r in rows], [
            "never_released_during_live_episode", "success", "disqualified_by_forbidden_contact",
            "disqualified_by_hand_penetration", "released_then_floor_termination",
            "released_but_no_final_paired_grasp", "released_but_insufficient_final_clearance",
            "released_but_insufficient_final_hold_duration"])
        self.assertFalse(rows[4]["forbidden_ever"])
        self.assertFalse(rows[4]["success"])
        self.assertEqual(rows[4]["peak_force_n"], 0)
        self.assertAlmostEqual(rows[4]["termination_time_s"], 2.55)


if __name__ == "__main__":
    unittest.main()
