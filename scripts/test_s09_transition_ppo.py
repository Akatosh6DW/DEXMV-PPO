"""Reproduce and repair logging/timeout loss without changing sampled states."""

import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_transition_ppo import transition_wrap
from mujoco_playground._src import mjx_env, wrapper


class TinyEnv:
    action_size = 1

    def __init__(self, failure):
        self.failure = failure

    def reset(self, rng):
        return mjx_env.State(data=jax.random.uniform(rng,()), obs=jp.zeros(1),
            reward=jp.float32(0), done=jp.float32(0), metrics=dict(score=jp.float32(0)),
            info=dict(task_step=jp.int32(0)))

    def step(self, state, action):
        n = state.info["task_step"]+1
        return state.replace(data=state.data+action[0], obs=jp.array([n],jp.float32),
            reward=jp.float32(1), done=(self.failure & (n>=2)).astype(jp.float32),
            metrics=dict(score=jp.float32(1)), info=dict(state.info, task_step=n))


class TransitionTests(unittest.TestCase):
    def check_case(self, failure):
        a = wrapper.wrap_for_brax_training(TinyEnv(failure),full_reset=True,episode_length=3)
        b = transition_wrap(TinyEnv(failure),episode_length=3)
        rng = jax.random.split(jax.random.PRNGKey(3),2)
        x,y = a.reset(rng),b.reset(rng)
        fa,fb = jax.jit(a.step),jax.jit(b.step)
        terminals=0
        for _ in range(8):
            x,y = fa(x,jp.ones((2,1))),fb(y,jp.ones((2,1)))
            for old,new in ((x.data,y.data),(x.obs,y.obs),(x.reward,y.reward),(x.done,y.done),
                            (x.info["task_step"],y.info["task_step"])):
                np.testing.assert_array_equal(old,new)
            if bool(y.done[0]):
                terminals += 1
                np.testing.assert_array_equal(x.info["episode_done"],0)
                np.testing.assert_array_equal(y.info["episode_done"],1)
                np.testing.assert_array_equal(y.info["truncation"],0 if failure else 1)
                np.testing.assert_array_equal(x.info["truncation"],0)
                length=2 if failure else 3
                np.testing.assert_array_equal(y.info["episode_metrics"]["length"],length)
                np.testing.assert_array_equal(y.info["episode_metrics"]["sum_reward"],length)
        self.assertGreaterEqual(terminals,2)

    def test_timeout_preserves_truncation_and_episode_metrics(self):
        self.check_case(False)

    def test_failure_stays_true_terminal_and_next_episode_resets(self):
        self.check_case(True)


if __name__ == "__main__":
    unittest.main()
