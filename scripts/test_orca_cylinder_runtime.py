"""Native cylinder control equivalence, observation boundaries, and push timing."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import jax
import jax.numpy as jp
import numpy as np
from orca_shape_pilot_env import OrcaShapePilot
from run_orca_suite_job import MODEL

REFERENCE = ROOT / "artifacts/orca_cylinder_study/D04_reference/reference.npz"


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pure = OrcaShapePilot(MODEL,"pure",REFERENCE,shape="cylinder",audit=True)
        cls.residual = OrcaShapePilot(MODEL,"residual",REFERENCE,shape="cylinder",audit=True)

    def test_absolute_teacher_targets_match_zero_residual(self):
        a=jax.jit(self.pure.reset)(jax.random.PRNGKey(98001))
        b=jax.jit(self.residual.reset)(jax.random.PRNGKey(98001))
        np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
        pure_step,res_step=jax.jit(self.pure.step),jax.jit(self.residual.step)
        for _ in range(65):
            action=self.pure.ctrl_to_action(self.residual.next_base(b.info["step"]))
            a,b=pure_step(a,action),res_step(b,jp.zeros(17))
            # Fused residual conversion and an external absolute action round
            # differently in float32. Compare commands, not chaotic trajectories.
            np.testing.assert_allclose(a.data.ctrl,b.data.ctrl,rtol=0,atol=5e-7)
            np.testing.assert_allclose(a.info["desired_ctrl"],b.info["desired_ctrl"],rtol=0,atol=5e-7)
            np.testing.assert_array_equal(a.obs["state"][95:112],0.)

    def test_identical_commands_give_identical_native_physics(self):
        a=jax.jit(self.pure.reset)(jax.random.PRNGKey(98003)).data
        b=jax.jit(self.residual.reset)(jax.random.PRNGKey(98003)).data
        left,right=jax.jit(self.pure._step_physics),jax.jit(self.residual._step_physics)
        for i in range(65):
            command=self.residual.reference_ctrl[i]
            a,b=left(a,command),right(b,command)
            np.testing.assert_array_equal(a.integration_words,b.integration_words)
            np.testing.assert_array_equal(a.substep_audit,b.substep_audit)

    def test_explicit_substep_force_timing(self):
        env=OrcaShapePilot(MODEL,"residual",REFERENCE,shape="cylinder",audit=True,benchmark="B5_push")
        state=jax.jit(env.reset)(jax.random.PRNGKey(98002))
        step=jax.jit(env.step)
        for index,expected in ((199,0.),(200,.15),(203,.15),(204,0.)):
            nxt=step(state.replace(info=dict(state.info,step=jp.int32(index))),jp.zeros(17))
            self.assertAlmostEqual(float(jp.linalg.norm(nxt.info["applied_push"])),expected,places=6)

    def test_position_perturbation_changes_only_reset_distribution(self):
        env=OrcaShapePilot(MODEL,"residual",REFERENCE,shape="cylinder",audit=True,benchmark="B1_position")
        self.assertEqual(env._config.spawn_config.cube_jitter,[.006,.006,0.])
        for key in ("body_mass","body_inertia","actuator_ctrlrange","geom_friction"):
            np.testing.assert_array_equal(getattr(env.mj_model,key),getattr(self.residual.mj_model,key))


if __name__ == "__main__":
    unittest.main()
