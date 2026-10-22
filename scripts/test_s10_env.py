"""S10 callback isolation, common observations and residual action semantics."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Aero-Hand/sim_rl/mujoco_playground"))

import jax
import jax.numpy as jp
import numpy as np

from orca_s10_env import OrcaS10Grasp
from s10_mobile_base import OUT


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selection=json.loads((OUT/"inputs/selection.json").read_text())
        cls.pure=OrcaS10Grasp(OUT/"inputs/model.xml",selection)
        cls.residual=OrcaS10Grasp(OUT/"inputs/model.xml",selection,arm="residual")

    def test_common_observations(self):
        a=self.pure.reset(jax.random.PRNGKey(3))
        b=self.residual.reset(jax.random.PRNGKey(3))
        self.assertEqual(a.obs["state"].shape,(123,))
        self.assertEqual(a.obs["privileged_state"].shape,(140,))
        for key in a.obs:
            np.testing.assert_array_equal(a.obs[key],b.obs[key])

    def test_interleaved_episodes_restore_plan(self):
        env=self.pure
        a=env.reset(jax.random.PRNGKey(5))
        expected=env.step(a,env.ctrl_to_action(env.reference_base(a.info)))
        b=env.reset(jax.random.PRNGKey(73))
        env.step(b,jp.zeros(17))
        actual=env.step(a,env.ctrl_to_action(env.reference_base(a.info)))
        np.testing.assert_array_equal(expected.data.integration_words,actual.data.integration_words)
        np.testing.assert_array_equal(expected.obs["state"],actual.obs["state"])

    def test_zero_residual_matches_reference(self):
        pure,residual=self.pure,self.residual
        a=pure.reset(jax.random.PRNGKey(8))
        b=residual.reset(jax.random.PRNGKey(8))
        stepa=jax.jit(pure.step); stepb=jax.jit(residual.step)
        for _ in range(110):
            a=stepa(a,pure.ctrl_to_action(pure.reference_base(a.info)))
            b=stepb(b,jp.zeros(17))
        # Absolute-action normalization incurs float32 rounding, unlike direct residual addition.
        np.testing.assert_allclose(a.data.qpos,b.data.qpos,atol=2e-4,rtol=0)
        np.testing.assert_allclose(a.info["desired_ctrl"],b.info["desired_ctrl"],atol=2e-7,rtol=0)
        np.testing.assert_array_equal(b.info["requested_residual"],0)
        self.assertEqual(float(b.metrics["rope_force_n"]),0.)


if __name__=="__main__":
    unittest.main()
