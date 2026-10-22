"""Only newly enabled actor input connections receive smaller Adam updates."""

import unittest

import jax
import jax.numpy as jp
import numpy as np
import optax
from brax.training.agents.ppo.losses import PPONetworkParams

from run_s09_slow_feedback import scale_feedback_updates
from run_s09_constrained_ppo import split_adam
from run_s09_bc_phase_curriculum import KEEP


class FeedbackTests(unittest.TestCase):
    def test_selective_adam_updates_leave_head_and_critic_unchanged(self):
        params=PPONetworkParams(policy={"params":{
            "hidden_0":{"kernel":jp.ones((123,3)),"bias":jp.ones(3)},
            "hidden_2":{"kernel":jp.ones((3,34)),"bias":jp.ones(34)}}},value={"w":jp.ones(4)})
        original=split_adam(3e-6,3e-4)
        scaled=optax.chain(split_adam(3e-6,3e-4),scale_feedback_updates(.1))
        a,b=original.init(params),scaled.init(params)
        for i in range(4):
            gradients=jax.tree_util.tree_map(lambda x:x*(i+1),params)
            x,a=jax.jit(original.update)(gradients,a)
            y,b=jax.jit(scaled.update)(gradients,b)
            kernel=x.policy["params"]["hidden_0"]["kernel"]
            changed=y.policy["params"]["hidden_0"]["kernel"]
            np.testing.assert_allclose(changed[KEEP],kernel[KEEP],rtol=1e-6)
            np.testing.assert_allclose(changed[~KEEP],kernel[~KEEP]*.1,rtol=1e-6)
            np.testing.assert_array_equal(x.value["w"],y.value["w"])
            np.testing.assert_array_equal(x.policy["params"]["hidden_0"]["bias"],y.policy["params"]["hidden_0"]["bias"])
            for name in ("kernel","bias"):
                np.testing.assert_array_equal(x.policy["params"]["hidden_2"][name],y.policy["params"]["hidden_2"][name])

    def test_does_not_silently_freeze_feedback(self):
        for scale in (0,-1,1.1,float("nan")):
            with self.assertRaises(ValueError):
                scale_feedback_updates(scale)


if __name__ == "__main__":
    unittest.main()
