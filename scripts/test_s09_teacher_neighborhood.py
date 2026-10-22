import unittest

import jax
import jax.numpy as jp
import numpy as np

from probe_s09_teacher_neighborhood import prefix_factory


class TeacherNeighborhoodTests(unittest.TestCase):
    def test_prefix_uses_original_action_at_elapsed_step(self):
        actions = jp.arange(800 * 17, dtype=jp.float32).reshape(800, 17)
        obs = jp.zeros((4, 123)).at[:, 112].set(jp.array([0., 39 / 800, 40 / 800, 1.1]))
        policy = prefix_factory(actions)(None)
        action, _ = jax.jit(policy)(dict(state=obs), jax.random.PRNGKey(0))
        np.testing.assert_array_equal(action, actions[jp.array([0, 39, 40, 799])])

    def test_invalid_teacher_action_shape(self):
        with self.assertRaises(ValueError):
            prefix_factory(np.zeros((40, 17)))


if __name__ == "__main__":
    unittest.main()
