"""Shared reset/physics, reference ablation, intervention and substep tests."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from orca_acquisition_env import OrcaAcquisitionGrasp
from orca_suite_env import OrcaSuiteGrasp, AuditedPhysics
from run_orca_suite_job import MODEL, REFERENCE, RAW_REFERENCE, initial_gate


class SuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_pmap_shmap_merge", False)
        cls.original = OrcaAcquisitionGrasp(MODEL, "residual", REFERENCE)
        cls.base = OrcaSuiteGrasp(MODEL, "residual", REFERENCE, audit=True)

    def test_nominal_dynamics_match_original(self):
        reset0, reset1 = jax.jit(self.original.reset), jax.jit(self.base.reset)
        a, b = reset0(jax.random.PRNGKey(331)), reset1(jax.random.PRNGKey(331))
        step0, step1 = jax.jit(self.original.step), jax.jit(self.base.step)
        np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        for _ in range(20):
            a, b = step0(a, jp.zeros(17)), step1(b, jp.zeros(17))
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
            np.testing.assert_array_equal(a.obs["state"], b.obs["state"])
            self.assertAlmostEqual(float(a.reward), float(b.reward))

    def test_variants_keep_initial_state(self):
        state = jax.jit(self.base.reset)(jax.random.PRNGKey(332))
        for variant in ("linear", "unrefined"):
            env = OrcaSuiteGrasp(MODEL, "residual", REFERENCE, variant=variant, raw_reference=RAW_REFERENCE)
            other = jax.jit(env.reset)(jax.random.PRNGKey(332))
            np.testing.assert_array_equal(state.data.qpos, other.data.qpos)
            self.assertFalse(np.array_equal(env.reference_ctrl, self.base.reference_ctrl))
            if variant == "linear":
                np.testing.assert_allclose(env.reference_ctrl[-1], self.base.reference_ctrl[-1])
                np.testing.assert_allclose(env.reference_ctrl[29], self.base.reference_ctrl[-1])

    def test_perturbation_configuration(self):
        for name in ("B2_yaw", "B3_friction", "B4_mass", "B5_push"):
            env = OrcaSuiteGrasp(MODEL, "residual", REFERENCE, benchmark=name)
            state = jax.jit(env.reset)(jax.random.PRNGKey(333))
            if name == "B2_yaw":
                self.assertFalse(np.allclose(state.data.qpos[env._cube_qids[3:]], self.base.object_quaternion))
                np.testing.assert_allclose(np.linalg.norm(state.data.qpos[env._cube_qids[3:]]), 1, atol=1e-6)
            if name == "B3_friction":
                mask = env.mj_model.geom_contype != 0
                np.testing.assert_allclose(env.mj_model.geom_friction[mask, 0],
                                           self.base.mj_model.geom_friction[mask, 0]*.6)
            if name == "B4_mass":
                self.assertAlmostEqual(env.mj_model.body("cube").mass[0], .07)
            if name == "B5_push":
                step = jax.jit(env.step)
                for index, expected in ((199, 0.), (200, .15), (203, .15), (204, 0.)):
                    changed = step(state.replace(info=dict(state.info, step=jp.int32(index))), jp.zeros(17))
                    self.assertAlmostEqual(float(jp.linalg.norm(changed.info["applied_push"])), expected, places=6)

    def test_zero_gain_is_zero_correction(self):
        env = OrcaSuiteGrasp(MODEL, "residual", REFERENCE, residual_gain=0.)
        state = jax.jit(env.reset)(jax.random.PRNGKey(334))
        nxt = jax.jit(env.step)(state, jp.ones(17))
        np.testing.assert_array_equal(nxt.info["requested_residual"], np.zeros(17))

    def test_all_benchmark_reset_gates(self):
        from orca_suite_env import BENCHMARKS
        for name in BENCHMARKS:
            env = OrcaSuiteGrasp(MODEL, "pure", REFERENCE, benchmark=name)
            self.assertTrue(initial_gate(env, 64)["passed"], name)

    def test_substep_audit_records_interior_peak(self):
        physics = self.base.native
        state = jax.jit(self.base.reset)(jax.random.PRNGKey(335))
        counts = []
        original = physics.sample_contacts
        def sample(data):
            original(data)
            counts.append(1)
            if len(counts) == 3:
                physics.audit[0] = .004
        physics.sample_contacts = sample
        try:
            data = physics.single("forced_step", np.asarray(state.data.integration_words),
                                  np.asarray(state.data.ctrl), np.asarray(state.data.mocap_pos), np.zeros(3))
        finally:
            physics.sample_contacts = original
        self.assertEqual(len(counts), 10)
        self.assertGreaterEqual(data.substep_audit[0], .004-1e-8)


if __name__ == "__main__":
    unittest.main()
