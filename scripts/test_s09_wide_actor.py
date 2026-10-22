import copy
import unittest

import jax
import numpy as np

from run_s09_wide_actor import widen_actor, WIDTH


class WideActorTests(unittest.TestCase):
    def actor(self):
        rng = np.random.default_rng(12)
        return dict(params={f"hidden_{i}": dict(
            kernel=rng.normal(size=(a,b)).astype(np.float32), bias=rng.normal(size=b).astype(np.float32))
            for i,(a,b) in enumerate(((5,3),(3,3),(3,2)))})

    def test_function_and_source_preserved(self):
        actor = self.actor()
        saved = copy.deepcopy(actor)
        expanded, info = widen_actor(actor, width=11)
        x = np.random.default_rng(4).normal(size=(20,5))
        def apply(params):
            y = x
            for i in range(3):
                layer = params["params"][f"hidden_{i}"]
                y = y@np.asarray(layer["kernel"])+np.asarray(layer["bias"])
                if i < 2:
                    y = np.tanh(y)
            return y
        np.testing.assert_allclose(apply(actor), apply(expanded), atol=1e-6, rtol=0)
        for a,b in zip(jax.tree_util.tree_leaves(actor), jax.tree_util.tree_leaves(saved)):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(info["new_width"], 11)
        np.testing.assert_array_equal(expanded["params"]["hidden_2"]["kernel"][3:], 0)
        np.testing.assert_array_equal(expanded["params"]["hidden_1"]["kernel"][3:,:3], 0)

    def test_new_units_receive_gradient(self):
        import jax.numpy as jp
        actor, _ = widen_actor(self.actor(), width=11)
        def loss(params):
            y = jp.ones((5,))
            for i in range(3):
                layer = params["params"][f"hidden_{i}"]
                y = y@layer["kernel"]+layer["bias"]
                if i < 2:
                    y = jp.tanh(y)
            return jp.sum(y)
        gradient = jax.grad(loss)(actor)
        self.assertGreater(float(jp.linalg.norm(gradient["params"]["hidden_2"]["kernel"][3:])), 0)

    def test_reject_shrinking(self):
        with self.assertRaises(ValueError):
            widen_actor(self.actor(), width=2)

    def test_near_sixfold_parameter_capacity(self):
        count = lambda n:123*n+n+n*n+n+n*34+34
        self.assertLess(abs(count(WIDTH)/(6*count(128))-1), .001)

    def test_matched_inference_shape(self):
        import jax.numpy as jp
        from brax.training.networks import FeedForwardNetwork
        from audit_s09_wide_actor import matched_actor
        base = FeedForwardNetwork(init=lambda _: None,
            apply=lambda _,p,obs: obs["state"]@p)
        actor = matched_actor(base, 8)
        obs = dict(state=jp.arange(4*7*5, dtype=jp.float32).reshape(4,7,5))
        params = jp.ones((5,2))
        np.testing.assert_array_equal(actor.apply(None, params, obs), base.apply(None, params, obs))
        with self.assertRaises(ValueError):
            matched_actor(base, 2).apply(None, params, obs)


if __name__ == "__main__":
    unittest.main()
