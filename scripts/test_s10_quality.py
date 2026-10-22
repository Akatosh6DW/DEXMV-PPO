"""Acquisition penetrations remain visible even when the old gate passes."""

import unittest

import numpy as np

from run_s10_quality_job import quality_summary


class QualityTests(unittest.TestCase):
    def test_pre_release_penetration_not_hidden(self):
        audit=np.zeros((1,2,13),np.float32)
        audit[0,:,6]=[.005,.001]
        audit[0,:,7]=[20.,2.]
        audit[0,1,10]=1
        trace=dict(active=np.ones((1,2),bool),alive=np.ones((1,2),bool),audit=audit,
            success30=np.ones((1,2),bool),success15=np.ones((1,2),bool),hold=np.full((1,2),30.),
            best_hold=np.full((1,2),30.),initial_penetration=np.zeros((1,2)),
            requested_residual=np.zeros((1,2,17)),reference_index=np.zeros((1,2)),
            qpos=np.zeros((1,2,31)),mocap=np.zeros((1,2,2,3)))
        summary,rows=quality_summary(trace,[1])
        self.assertTrue(rows[0]["success30"])
        self.assertFalse(rows[0]["full_path_success30"])
        self.assertEqual(summary["success30"],1.)
        self.assertEqual(summary["full_path_success30"],0.)
        self.assertEqual(rows[0]["peak_acquisition_force_n"],20.)
        self.assertEqual(rows[0]["peak_released_force_n"],2.)


if __name__=="__main__":
    unittest.main()
