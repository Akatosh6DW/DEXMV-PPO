"""Head-only PPO must preserve hidden weights and the critic optimizer."""

import copy
import unittest

import jax
import jax.numpy as jp
import numpy as np
import optax
from brax.training.agents.ppo.losses import PPONetworkParams

from run_s09_constrained_ppo import split_adam
from run_s09_head_only import head_only_gradients, verify_frozen_hidden


class HeadOnlyTests(unittest.TestCase):
    def actor(self):
        return {"params": {f"hidden_{i}": {"kernel": jp.ones(shape), "bias": jp.ones(shape[1])}
                           for i, shape in enumerate(((123, 3), (3, 3), (3, 34)))}}

    def test_update_scope_and_critic(self):
        params = PPONetworkParams(policy=self.actor(), value={"w": jp.ones(4)})
        base = split_adam(3e-6, 3e-4)
        probe = optax.chain(head_only_gradients(), split_adam(3e-6, 3e-4))
        sa, sb = base.init(params), probe.init(params)
        for scale in (1., -2., .5, 4.):
            grads = jax.tree_util.tree_map(lambda x: x*scale, params)
            a, sa = base.update(grads, sa)
            b, sb = probe.update(grads, sb)
            np.testing.assert_array_equal(a.value["w"], b.value["w"])
            for name in ("hidden_0", "hidden_1"):
                for value in b.policy["params"][name].values():
                    np.testing.assert_array_equal(value, 0)
            for name in ("kernel", "bias"):
                np.testing.assert_array_equal(a.policy["params"]["hidden_2"][name],
                                              b.policy["params"]["hidden_2"][name])

    def test_audit(self):
        before = (None, self.actor())
        after = copy.deepcopy(before)
        after[1]["params"]["hidden_2"]["kernel"] += 1
        verify_frozen_hidden(before, after)
        after[1]["params"]["hidden_0"]["bias"] += 1
        with self.assertRaises(AssertionError):
            verify_frozen_hidden(before, after)


if __name__ == "__main__":
    unittest.main()
