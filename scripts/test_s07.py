"""Regression tests for the final compiled cylinder and bounded controls."""

from pathlib import Path
import unittest

import jax
import mujoco
import numpy as np

from orca_s06_env import OrcaS06Grasp
from orca_s07_env import OrcaS07Grasp
from orca_residual_controls import compose

INPUT = Path(__file__).resolve().parents[1] / "artifacts/orca_benchmark_suite/S06/P09_implicit"


class Tests(unittest.TestCase):
    def env(self, **kw):
        return OrcaS07Grasp(INPUT / "model.xml", "residual", INPUT / "reference.npz", **kw)

    def test_nominal_model_unchanged(self):
        original = OrcaS06Grasp(INPUT / "model.xml", "residual", INPUT / "reference.npz")
        new = self.env()
        for name in ("geom_friction", "body_mass", "body_inertia", "actuator_gainprm", "geom_size"):
            np.testing.assert_array_equal(getattr(new.mj_model, name), getattr(original.mj_model, name))
        self.assertEqual(new.mj_model.opt.integrator, mujoco.mjtIntegrator.mjINT_IMPLICIT)
        self.assertEqual(new.n_substeps, 10)

    def test_mass_and_friction_affect_final_model(self):
        base, mass, slip = self.env(), self.env(benchmark="B4_mass"), self.env(benchmark="B3_friction")
        bid = base.mj_model.body("cube").id
        np.testing.assert_allclose(mass.mj_model.body_inertia[bid], base.mj_model.body_inertia[bid]*2)
        self.assertAlmostEqual(mass.mj_model.body_mass[bid], .07)
        physical = base.mj_model.geom_contype != 0
        np.testing.assert_allclose(slip.mj_model.geom_friction[physical, 0],
                                   base.mj_model.geom_friction[physical, 0]*.6)

    def test_control_clips_are_nonintegrating(self):
        a, r = compose(np.array([.7]), np.array([2.]), .08, np.array([-1.]), np.array([1.]))
        np.testing.assert_allclose(a, [.78])
        np.testing.assert_allclose(r, [.08])
        b, _ = compose(np.array([.7]), np.array([2.]), .08, np.array([-1.]), np.array([1.]))
        np.testing.assert_array_equal(a, b)

    def test_zero_gain_matches_reference_controls(self):
        env = self.env(residual_gain=0.)
        state = env.reset(jax.random.PRNGKey(73000))
        nxt = env.step(state, np.ones(17, np.float32))
        np.testing.assert_allclose(nxt.info["requested_residual"], 0)
        np.testing.assert_allclose(nxt.data.ctrl, env.reference_ctrl[0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
