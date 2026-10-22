import unittest

import numpy as np

from diagnose_s09_gae_gradient_conflict import segmented_targets


class SegmentedTargetsTests(unittest.TestCase):
    def test_one_step_target_and_excluded_last_segment(self):
        values=np.arange(8,dtype=np.float32)[None,:]
        reward=np.ones_like(values)
        target=segmented_targets(reward,values,np.ones_like(values,bool),length=2,lambda_=0)
        self.assertEqual(target.shape,(1,6))
        np.testing.assert_allclose(target,1+.99*values[:,1:7],atol=1e-6)

    def test_termination_blocks_future_reward_and_bootstrap(self):
        reward=np.array([[1,2,999,999,999]],np.float32)
        values=np.ones_like(reward)*100
        active=np.array([[True,True,False,False,False]])
        target=segmented_targets(reward,values,active,length=2,lambda_=1)
        np.testing.assert_allclose(target[:,:2],[[1+.99*2,2]],atol=2e-5)

    def test_bootstrap_and_validation(self):
        values=np.ones((2,5),np.float32)*3
        reward=np.ones_like(values)
        target=segmented_targets(reward,values,np.ones_like(values,bool),length=2,lambda_=1)
        expected=np.array([[1+.99+(.99**2)*3,1+.99*3]*2]*2)
        np.testing.assert_allclose(target,expected,atol=1e-6)
        for active in (np.array([[True,False,True,False,False]]*2), np.ones((2,4),bool)):
            with self.assertRaises(ValueError):
                segmented_targets(reward,values,active)


if __name__=="__main__":
    unittest.main()
