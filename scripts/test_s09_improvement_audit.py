"""Reference provenance must be verified, not merely self-consistent controls."""

import unittest

import jax
import numpy as np

from audit_s09_ppo_improvement import verify_reference
from orca_s09_env import OrcaS09Grasp
from s09_suspension import OUT


class ReferenceBindingTests(unittest.TestCase):
    def test_wrong_base_or_mode_is_rejected(self):
        env=OrcaS09Grasp(OUT/"inputs/model.xml","residual",OUT/"inputs/reference.npz")
        seed=130001
        state=jax.jit(env.reset)(jax.random.PRNGKey(seed))
        index=int(state.info["reference_index"])
        trace=dict(seeds=np.array([seed]), active=np.array([[True,True,True,False]]),
            reference_index=np.full((1,4),index), video_base=np.asarray(env.reference_ctrl)[index,:4][None].copy())
        self.assertEqual(verify_reference(env,trace),0.)
        trace["video_base"][0,1,0]+=.001
        with self.assertRaises(AssertionError):
            verify_reference(env,trace)
        trace["video_base"]=np.asarray(env.reference_ctrl)[index,:4][None].copy()
        trace["reference_index"][0,0]=(index+1)%6
        with self.assertRaises(AssertionError):
            verify_reference(env,trace)


if __name__ == "__main__":
    unittest.main()
