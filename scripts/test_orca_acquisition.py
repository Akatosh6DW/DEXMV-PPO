"""Focused checks for shared physics, residual composition, and full resets."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src import wrapper
from orca_acquisition_env import OrcaAcquisitionGrasp
from train_orca_acquisition import MODEL, REFERENCE, initial_gate


class AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_pmap_shmap_merge", False)
        cls.pure = OrcaAcquisitionGrasp(MODEL, "pure", REFERENCE)
        cls.residual = OrcaAcquisitionGrasp(MODEL, "residual", REFERENCE)

    def test_same_reset_and_controls(self):
        p = jax.jit(self.pure.reset)(jax.random.PRNGKey(654))
        r = jax.jit(self.residual.reset)(jax.random.PRNGKey(654))
        np.testing.assert_array_equal(p.data.qpos, r.data.qpos)
        np.testing.assert_array_equal(p.data.mocap_pos, r.data.mocap_pos)
        pstep, rstep = jax.jit(self.pure.step), jax.jit(self.residual.step)
        for _ in range(12):
            base = self.residual.next_base(r.info["step"])
            p = pstep(p, self.pure.ctrl_to_action(base))
            r = rstep(r, jp.zeros(17))
            np.testing.assert_array_equal(p.data.qpos, r.data.qpos)
            np.testing.assert_allclose(p.reward, r.reward, atol=1e-6)
            np.testing.assert_allclose(r.info["desired_ctrl"], base, atol=1e-6)

    def test_valid_open_resets(self):
        gate = initial_gate(self.pure, 64)
        self.assertLessEqual(gate["initial_penetration_max_m"], .003)

    def test_full_reset_wrapper(self):
        env = wrapper.wrap_for_brax_training(self.residual, episode_length=4, full_reset=True)
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(111), 2))
        step = jax.jit(env.step)
        for i in range(12):
            state = step(state, jp.zeros((2, 17)))
            self.assertTrue(np.isfinite(state.data.qpos).all())
            self.assertTrue(np.isfinite(state.obs["state"]).all())
            if (i+1) % 4 == 0:
                np.testing.assert_array_equal(state.info["step"], 0)
                np.testing.assert_allclose(state.data.ctrl,
                                           np.tile(self.residual._config.initial_ctrl, (2, 1)), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
