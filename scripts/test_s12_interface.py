"""Regression tests for soft physics state isolation and residual semantics."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"Aero-Hand/sim_rl/mujoco_playground"))

import jax
import jax.numpy as jp
import numpy as np

from orca_s12_env import OrcaS12Grasp, SoftPhysics
from s12_soft import SoftConfig, SoftGrasp


class InterfaceTests(unittest.TestCase):
    def test_nonzero_baseline_applied_once(self):
        cfg = SoftConfig(timestep=.0002, hand_offset_z=.003)
        native = SoftPhysics(cfg)
        sim = SoftGrasp(cfg)
        offset = np.array([.001, -.001, .001])
        start = native.single("s12_reset", offset)
        sim.reset(offset=offset)
        np.testing.assert_allclose(start.mocap_pos, sim.data.mocap_pos, atol=3e-8)
        expected = sim.original_pos+[0., -.2, .003]+offset
        np.testing.assert_allclose(sim.home, expected, atol=1e-12)
        target = np.zeros(17)
        result = native.single("s12_step", start.integration_words, target, offset)
        sim.step(target)
        np.testing.assert_allclose(result.qpos, sim.data.qpos, atol=3e-8)

    def test_worker_batch_matches_serial(self):
        native = SoftPhysics(SoftConfig(timestep=.0002))
        offsets = np.array([[0., 0., 0.], [.001, -.001, 0.]])
        start = native.batched("s12_reset", offsets)
        desired = np.zeros((2,17))
        serial = native.batched("s12_step", start.integration_words, desired, offsets)
        native.enable_workers(2)
        try:
            parallel = native.batched("s12_step", start.integration_words, desired, offsets)
            for a,b in zip(jax.tree_util.tree_leaves(serial), jax.tree_util.tree_leaves(parallel)):
                np.testing.assert_array_equal(a,b)
        finally:
            native.pool.close()

    def test_interleaved_integration_restore(self):
        native = SoftPhysics(SoftConfig(timestep=.0002))
        offset = np.array([.001, -.001, .001])
        start = native.single("s12_reset", offset)
        target = np.zeros(17)
        first = native.single("s12_step", start.integration_words, target, offset)
        native.single("s12_reset", -offset)
        again = native.single("s12_step", start.integration_words, target, offset)
        for a,b in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(again)):
            np.testing.assert_array_equal(a,b)

    def test_zero_residual_equals_reference(self):
        pure = OrcaS12Grasp(jitter=0)
        residual = OrcaS12Grasp(arm="residual", jitter=0)
        p = pure.reset(jax.random.PRNGKey(0))
        r = residual.reset(jax.random.PRNGKey(0))
        self.assertEqual(p.obs["state"].shape, (93,))
        self.assertEqual(p.obs["privileged_state"].shape, (113,))
        for _ in range(3):
            p = pure.step(p, pure.ctrl_to_action(pure.reference_base(p.info)))
            r = residual.step(r, jp.zeros(17))
        np.testing.assert_allclose(p.data.qpos, r.data.qpos, atol=2e-7)
        np.testing.assert_array_equal(r.info["requested_residual"], np.zeros(17))

    def test_nonzero_reference_and_requested_correction(self):
        pure = OrcaS12Grasp(jitter=0)
        residual = OrcaS12Grasp(arm="residual",jitter=0)
        p = pure.reset(jax.random.PRNGKey(0))
        r = residual.reset(jax.random.PRNGKey(0))
        # Isolate the action mapping using a nonzero phase target. This is not
        # a valid time-consistent task rollout; full rollouts are audited separately.
        p.info["step"] = r.info["step"] = jp.int32(80)
        base = residual.reference_base(r.info)
        self.assertGreater(float(jp.max(jp.abs(base))),.5)
        p1 = pure.step(p,pure.ctrl_to_action(base))
        r1 = residual.step(r,jp.zeros(17))
        np.testing.assert_allclose(p1.data.ctrl,r1.data.ctrl,atol=2e-7)
        changed = residual.step(r,jp.full(17,.5))
        np.testing.assert_allclose(changed.info["requested_residual"],.04,atol=1e-8)
        expected = jp.clip(base+.04,residual._lowers,residual._uppers)
        np.testing.assert_allclose(changed.info["desired_ctrl"],expected,atol=2e-7)


if __name__ == "__main__":
    unittest.main()
