"""Check showcase cut points without changing or running training."""

import unittest

import numpy as np

from export_grasp_presentation_videos import clip_endpoint, four_second_endpoint


def trace(length=200):
    out = {k: np.ones(length) for k in ("hold", "alive", "active", "released", "thumb")}
    out.update(contacts=np.full(length, 2), forbidden=np.zeros(length),
               substep_forbidden=np.zeros(length), substep_penetration=np.zeros(length),
               qpos=np.zeros((length, 24)))
    return out


class ClipEndpoints(unittest.TestCase):
    def test_full_four_seconds_between_first_and_last_valid_states(self):
        t = trace()
        t["released"][:59] = 0
        self.assertEqual(four_second_endpoint(t), (139, 59))

    def test_interruption_restarts_timer(self):
        t = trace()
        t["hold"][70] = 0
        self.assertEqual(four_second_endpoint(t), (151, 71))

    def test_failed_episode_is_not_cut_at_four_seconds(self):
        t = trace()
        t["alive"][150:] = 0
        end, start, reason, status = clip_endpoint(t, False)
        self.assertEqual(end, 150)
        self.assertIsNone(start)
        self.assertEqual(status, "失败")
        self.assertEqual(reason, "回合失败终止")

    def test_first_released_audit_failure_precedes_later_drop(self):
        t = trace()
        t["released"][:20] = 0
        t["substep_penetration"][[5, 95]] = .004
        t["alive"][160:] = 0
        self.assertEqual(clip_endpoint(t, False)[0], 95)

    def test_failed_but_not_terminated_uses_deadline(self):
        t = trace()
        t["hold"][:] = 0
        self.assertEqual(clip_endpoint(t, False)[0], 199)
        with self.assertRaises(ValueError):
            clip_endpoint(t, True)


if __name__ == "__main__":
    unittest.main()
