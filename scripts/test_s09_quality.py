"""Extra quality statistics must not rewrite the primary success criterion."""

import unittest

import numpy as np

from evaluate_s09_quality import quality_episodes


class QualityTests(unittest.TestCase):
    def test_pre_release_and_padding_are_distinguished(self):
        audit=np.zeros((1,4,13))
        audit[0,:,6]=[.004,.001,.002,1]
        audit[0,:,7]=[2,3,4,999]
        audit[0,1:,10]=1
        audit[0,1:3,11]=1
        audit[0,1:3,0]=[.03,.3]
        audit[0,1:3,1]=[.03,.3]
        original=[dict(seed=1,success30=True)]
        trace=dict(active=np.array([[True,True,True,False]]),audit=audit,
            effective_residual=np.zeros((1,4,17)))
        result=quality_episodes(trace,original)[0]
        self.assertTrue(result["success30"])
        self.assertFalse(result["success30_all_depth_le_3mm"])
        self.assertEqual(result["max_all_penetration_m"],.004)
        self.assertEqual(result["peak_pre_release_force_n"],2)
        self.assertEqual(result["peak_released_force_n"],4)
        self.assertAlmostEqual(result["p95_released_control_peak_force_n"],3.95)
        self.assertEqual(result["valid_contact_load_below_0_2n_fraction"],.5)
        self.assertEqual(original,[dict(seed=1,success30=True)])

    def test_no_valid_contact_uses_null_not_nan(self):
        trace=dict(active=np.array([[True,False]]),audit=np.zeros((1,2,13)),
            effective_residual=np.zeros((1,2,17)))
        result=quality_episodes(trace,[dict(seed=1,success30=False)])[0]
        self.assertIsNone(result["p95_released_control_peak_force_n"])
        self.assertIsNone(result["valid_contact_load_below_0_2n_fraction"])
        self.assertEqual(result["peak_released_force_n"],0)

    def test_streak_breaks_exclude_padding(self):
        audit=np.zeros((1,5,13))
        audit[0,:,10]=1
        audit[0,:,11]=[0,1,0,1,0]
        trace=dict(active=np.array([[True,True,True,True,False]]),audit=audit,
            effective_residual=np.zeros((1,5,17)))
        result=quality_episodes(trace,[dict(seed=1,success30=False)])[0]
        self.assertEqual(result["valid_streak_break_events"],1)
        self.assertAlmostEqual(result["total_valid_grasp_sec"],.1)

    def test_effective_target_correction_excludes_padding_and_preserves_request(self):
        trace=dict(active=np.array([[True,False]]),audit=np.zeros((1,2,13)),
            effective_residual=np.array([[[.002]*17,[99.]*17]]))
        original=[dict(seed=1,success30=False,residual_rms_rad=.01)]
        result=quality_episodes(trace,original)[0]
        self.assertAlmostEqual(result["effective_target_residual_rms_rad"],.002)
        self.assertAlmostEqual(result["max_effective_target_residual_rad"],.002)
        self.assertEqual(result["residual_rms_rad"],.01)
        self.assertNotIn("effective_target_residual_rms_rad",original[0])


if __name__ == "__main__":
    unittest.main()
