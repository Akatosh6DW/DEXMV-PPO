"""Separate Adam rates must not change the actor update at equal actor LR."""

import unittest

import jax
import jax.numpy as jp
import numpy as np
import optax
from brax.training.agents.ppo.losses import PPONetworkParams

from run_s09_constrained_ppo import OptimizerProxy, split_adam


class SplitOptimizerTests(unittest.TestCase):
    def test_only_critic_update_changes(self):
        params = PPONetworkParams(policy={"w":jp.ones(3)}, value={"w":jp.ones(2)})
        original, split = optax.adam(3e-6), split_adam(3e-6, 3e-4)
        original_state, split_state = original.init(params), split.init(params)
        for i in range(4):
            grads = jax.tree_util.tree_map(lambda x: x*(i+1), params)
            old, original_state = original.update(grads, original_state)
            new, split_state = jax.jit(split.update)(grads, split_state)
            np.testing.assert_allclose(old.policy["w"], new.policy["w"], rtol=1e-6)
            np.testing.assert_allclose(old.value["w"]*100, new.value["w"], rtol=1e-6)

    def test_proxy_does_not_mutate_optax(self):
        adam = optax.adam
        proxy = OptimizerProxy(optax, 3e-4)
        self.assertIs(proxy.apply_updates, optax.apply_updates)
        proxy.adam(3e-6)
        self.assertIs(optax.adam, adam)


if __name__ == "__main__":
    unittest.main()
