import unittest

import jax
import jax.numpy as jp
import numpy as np

import run_s09_residual_critic_experts as route


class CriticExpertsTests(unittest.TestCase):
    def test_identical_initial_functions_and_routed_value_gradients(self):
        learned = route.experts.learned
        base_action = lambda obs: jp.tanh(obs[..., :17])
        shapes = dict(state=(123,), privileged_state=(140,))
        standard = learned.BASE_FACTORY(shapes, 17)
        base = (None, None, standard.value_network.init(jax.random.PRNGKey(4)))
        outputs = []
        for count in (1, 6):
            with route.critic_experts(count), route.experts.residual_experts():
                factory = learned.residual_factory(base_action)
                outputs.append(learned.initialize_residual(factory, base, 9, .03))
        shared, p0 = outputs[0]
        routed, p6 = outputs[1]
        for a, b in zip(jax.tree_util.tree_leaves(p0[1]), jax.tree_util.tree_leaves(p6[1])):
            np.testing.assert_array_equal(a, b)
        obs = dict(state=jp.ones((6, 123)).at[:, 95:101].set(jp.eye(6)),
                   privileged_state=jp.ones((6, 140)).at[:, 95:101].set(jp.eye(6)))
        np.testing.assert_array_equal(shared.value_network.apply(p0[0], p0[2], obs),
                                      routed.value_network.apply(p6[0], p6[2], obs))
        np.testing.assert_array_equal(shared.policy_network.apply(p0[0], p0[1], obs),
                                      routed.policy_network.apply(p6[0], p6[1], obs))
        for mode in range(6):
            single = jax.tree_util.tree_map(lambda x: x[mode], obs)
            grad = jax.grad(lambda p: routed.value_network.apply(p6[0], p, single))(p6[2])
            for index, expert in enumerate(grad["experts"]):
                total = sum(float(jp.sum(jp.abs(x))) for x in jax.tree_util.tree_leaves(expert))
                if index == mode:
                    self.assertGreater(total, 0)
                else:
                    self.assertEqual(total, 0)

    def test_invalid_count_and_restoration(self):
        before = route.experts.residual_expert_factory, route.experts.SINGLE_INITIALIZE, route.experts.ARCHITECTURE
        with self.assertRaises(ValueError), route.critic_experts(2):
            pass
        with self.assertRaisesRegex(RuntimeError, "test"), route.critic_experts(6):
            raise RuntimeError("test")
        self.assertEqual(before, (route.experts.residual_expert_factory, route.experts.SINGLE_INITIALIZE,
                                  route.experts.ARCHITECTURE))
        with route.critic_experts(1):
            self.assertEqual(before, (route.experts.residual_expert_factory, route.experts.SINGLE_INITIALIZE,
                                      route.experts.ARCHITECTURE))


if __name__ == "__main__":
    unittest.main()
