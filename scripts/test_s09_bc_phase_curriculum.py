"""BC curriculum cannot accidentally rely on untrained physical-state weights."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_bc_phase_curriculum import KEEP, phase_initialization
from train_orca_matched import initialize_matched


class PhaseTests(unittest.TestCase):
    def test_physical_inputs_have_zero_initial_effect_but_remain_parameters(self):
        net, params = initialize_matched(SimpleNamespace(arm="residual"), 71,.03)
        # Use a nonzero output head so this test covers the first-layer mask.
        head = params[1]["params"]["hidden_2"]
        head["kernel"] = jp.ones_like(head["kernel"])*.02
        fixed = phase_initialization(params)
        obs = jax.random.normal(jax.random.PRNGKey(11),(8,123))
        x = net.policy_network.apply(fixed[0],fixed[1],{"state":obs})
        y = net.policy_network.apply(fixed[0],fixed[1],{"state":obs*jp.asarray(KEEP)})
        np.testing.assert_array_equal(x,y)
        gradients = jax.grad(lambda policy: jp.sum(net.policy_network.apply(
            fixed[0],policy,{"state":obs})))(fixed[1])
        self.assertGreater(float(jp.max(jp.abs(gradients["params"]["hidden_0"]["kernel"][~KEEP]))),0)


if __name__ == "__main__":
    unittest.main()
