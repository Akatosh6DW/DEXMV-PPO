"""Stateless native callbacks preserve full state and the shared task interface."""

from pathlib import Path
import unittest

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from orca_native_physics import OrcaNativeMatchedGrasp, STATE_SPEC

ROOT = Path(__file__).resolve().parents[1] / "artifacts/orca_collision_refit/O20_hic53_elliptic"


class NativePhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ROOT.exists():
            raise unittest.SkipTest("Matched model not available")
        cls.env = OrcaNativeMatchedGrasp(ROOT / "cube.xml", "residual", ROOT / "reference.npz", tactile=True)

    def test_integration_precision_and_stateless_step(self):
        physics = self.env.native
        data = mujoco.MjData(physics.model)
        data.qpos[self.env._hand_qids] = np.array(self.env._config.initial_ctrl)
        data.qpos[self.env._cube_qids] = np.r_[np.array(self.env._spawn_pos), np.array(self.env.object_quaternion)]
        data.ctrl[:] = np.array(self.env._config.initial_ctrl)
        data.mocap_pos[0] = np.array(self.env._spawn_pos) - [0, 0, self.env.half_height + .004]
        mujoco.mj_forward(physics.model, data)
        state = physics.pack(data)
        packed = np.empty(physics.state_size)
        mujoco.mj_getState(physics.model, data, packed, STATE_SPEC)
        np.testing.assert_array_equal(state.integration_words.view(np.float64), packed)
        for _ in range(physics.substeps):
            mujoco.mj_step(physics.model, data)
        expected = physics.pack(data)
        actual = physics.single("step", state.integration_words, state.ctrl, state.mocap_pos)
        # The callback accepts float32 commands/positions, as does the actor.
        baseline = mujoco.MjData(physics.model)
        mujoco.mj_setState(physics.model, baseline, packed, STATE_SPEC)
        baseline.ctrl[:], baseline.mocap_pos[:] = state.ctrl, state.mocap_pos
        for _ in range(physics.substeps):
            mujoco.mj_step(physics.model, baseline)
        expected = physics.pack(baseline)
        for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_array_equal(a, b)
        repeated = physics.single("step", state.integration_words, state.ctrl, state.mocap_pos)
        np.testing.assert_array_equal(actual.integration_words, repeated.integration_words)

    def test_jit_vmap_and_tactile_mask(self):
        env = self.env
        reset = jax.jit(jax.vmap(env.reset))
        keys = jp.stack([jax.random.PRNGKey(4)] * 2)
        state = reset(keys)
        self.assertEqual(state.obs["state"].shape, (2, 123))
        self.assertEqual(state.obs["privileged_state"].shape, (2, 140))
        state.info["tactile_actor_mask"] = jp.array([False, True])
        step = jax.jit(jax.vmap(env.step))
        state = step(state, jp.zeros((2, 17)))
        for name in ("qpos", "qvel", "integration_words", "finger_force"):
            np.testing.assert_array_equal(getattr(state.data, name)[0], getattr(state.data, name)[1])
        np.testing.assert_array_equal(state.obs["state"][:, 94], 0.)
        self.assertEqual(state.data.integration_words.dtype, jp.uint32)


if __name__ == "__main__":
    unittest.main()
