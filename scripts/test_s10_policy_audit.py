"""Only the diagnostic reward may have FP32-ulp differences, never actions."""

import unittest

from audit_s10_policy_replay import comparison_passes


class PolicyAuditTests(unittest.TestCase):
    def test_reward_only_roundoff(self):
        self.assertTrue(comparison_passes(dict(reward=1.5e-8,qpos=0.,action=0.,success30=0.)))
        self.assertFalse(comparison_passes(dict(reward=1e-5,qpos=0.,action=0.)))

    def test_no_physics_or_policy_tolerance(self):
        for key in ("qpos","qvel","action","audit","obs","success30"):
            self.assertFalse(comparison_passes({key:1e-12,"reward":0.}))


if __name__=="__main__":
    unittest.main()
