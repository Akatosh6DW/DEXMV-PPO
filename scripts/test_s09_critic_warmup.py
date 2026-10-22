"""Warmup freezes the actor exactly, including optimizer momentum."""

import unittest

import jax
import jax.numpy as jp
import numpy as np
import optax
from brax.training.agents.ppo.losses import PPONetworkParams

from run_s09_critic_warmup import delayed_adam, warmup_optimizer


class WarmupTests(unittest.TestCase):
    def test_first_active_step_has_no_warmup_momentum(self):
        params={"w":jp.ones(3)}
        optimizer=delayed_adam(3e-5,2)
        state=optimizer.init(params)
        update=jax.jit(optimizer.update)
        for _ in range(2):
            delta,state=update({"w":jp.ones(3)*-100},state)
            np.testing.assert_array_equal(delta["w"],0)
        delta,state=update({"w":jp.ones(3)},state)
        plain=optax.adam(3e-5)
        expected,_=plain.update({"w":jp.ones(3)},plain.init(params))
        np.testing.assert_allclose(delta["w"],expected["w"],rtol=1e-6)

    def test_critic_updates_during_actor_freeze(self):
        params=PPONetworkParams(policy={"w":jp.ones(3)},value={"w":jp.ones(2)})
        optimizer=warmup_optimizer(3e-6,3e-4,2)
        state=optimizer.init(params)
        update=jax.jit(optimizer.update)
        for i in range(3):
            delta,state=update(jax.tree_util.tree_map(jp.ones_like,params),state)
            if i<2:
                np.testing.assert_array_equal(delta.policy["w"],0)
            else:
                self.assertGreater(float(jp.linalg.norm(delta.policy["w"])),0)
            self.assertGreater(float(jp.linalg.norm(delta.value["w"])),0)


if __name__ == "__main__":
    unittest.main()
