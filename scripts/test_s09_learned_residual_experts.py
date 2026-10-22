import unittest

import jax
import jax.numpy as jp
import numpy as np

import run_s09_learned_base_residual as learned
from run_s09_learned_residual_experts import residual_experts


class ResidualExpertsTests(unittest.TestCase):
    def test_initial_functions_and_gradient_routing(self):
        action = lambda obs:jp.tanh(obs[..., :17])
        shapes = dict(state=(123,),privileged_state=(140,))
        standard = learned.BASE_FACTORY(shapes,17)
        base_params = (None,None,standard.value_network.init(jax.random.PRNGKey(4)))
        factory = learned.residual_factory(action)
        single,params0 = learned.initialize_residual(factory,base_params,9,.03)
        before = learned.residual_factory
        with residual_experts():
            factory = learned.residual_factory(action)
            routed,params = learned.initialize_residual(factory,base_params,9,.03)
        self.assertIs(learned.residual_factory,before)
        for mode in range(6):
            obs = dict(state=jp.zeros(123).at[95+mode].set(1),privileged_state=jp.ones(140))
            np.testing.assert_array_equal(routed.policy_network.apply(params[0],params[1],obs),
                single.policy_network.apply(params0[0],params0[1],obs))
            grad = jax.grad(lambda p:jp.sum(routed.policy_network.apply(params[0],p,obs)[...,:17]))(params[1])
            for i,expert in enumerate(grad["experts"]):
                total = sum(float(jp.sum(jp.abs(x))) for x in jax.tree_util.tree_leaves(expert))
                if i == mode:
                    self.assertGreater(total,0)
                else:
                    self.assertEqual(total,0)
        for a,b in zip(jax.tree_util.tree_leaves(params0[2]),jax.tree_util.tree_leaves(params[2])):
            np.testing.assert_array_equal(a,b)


if __name__ == "__main__":
    unittest.main()
