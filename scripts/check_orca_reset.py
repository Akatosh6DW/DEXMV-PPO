"""GPU smoke test: an early drop must reset task clocks and support state."""

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src import wrapper
from orca_benchmark_env import OrcaBenchmark


env = OrcaBenchmark()
wrapped = wrapper.wrap_for_brax_training(env, episode_length=800, full_reset=True)
state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(3), 2))
qpos = state.data.qpos.at[:, env._cube_qids[2]].set(.01)
state = state.replace(data=state.data.replace(qpos=qpos))
state.info["step"] = jp.full((2,), 500, dtype=jp.int32)
state = jax.jit(wrapped.step)(state, jp.zeros((2, 17)))
assert np.all(np.asarray(state.done) == 1), state.done
assert np.all(np.asarray(state.info["step"]) == 0), state.info["step"]
assert np.all(np.asarray(state.info["stable_hold_steps"]) == 0)
np.testing.assert_allclose(np.asarray(state.data.mocap_pos[:, 0, 2]), .243, atol=1e-6)
print("PASS: early drop restores support, acquisition clock, and hold counter")
