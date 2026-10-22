"""Projection is bounded and leaves the original actor inputs/output visible."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax.numpy as jp
import numpy as np

from probe_orca_reference_tube import correction_metrics, projected_actor


class ReferenceTubeTests(unittest.TestCase):
    def test_push_applies_world_force_not_torque(self):
        import mujoco
        from evaluate_orca_matched_native import push_at_step
        model = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".01" gravity="0 0 0"/>'
                                              '<worldbody><body name="cube"><freejoint/>'
                                              '<geom type="box" size=".01 .01 .01" mass="1"/>'
                                              '</body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        data.xfrc_applied[model.body("cube").id, :3] = push_at_step(240, .1)
        for _ in range(20):
            mujoco.mj_step(model, data)
        np.testing.assert_allclose(data.qvel, [.02, 0, 0, 0, 0, 0], atol=1e-12)

    def test_push_schedule_has_exact_pulse_duration_and_zero_default(self):
        from evaluate_orca_matched_native import push_at_step
        pushes = np.stack([push_at_step(i, .1) for i in range(800)])
        np.testing.assert_array_equal(np.flatnonzero(pushes[:, 0] > 0), np.arange(240, 244))
        np.testing.assert_array_equal(np.flatnonzero(pushes[:, 0] < 0), np.arange(480, 484))
        np.testing.assert_array_equal(pushes[:, 1:], 0.)
        np.testing.assert_allclose(pushes.sum(axis=0), 0., atol=1e-15)
        np.testing.assert_array_equal(push_at_step(240, 0.), 0.)

    def test_bound_and_slew_limit(self):
        env = SimpleNamespace(_lowers=-jp.ones(17), _uppers=jp.ones(17), ctrl_to_action=lambda x: x)
        def original(*_):
            return jp.ones(123), jp.full(17, .3), None, None, None, jp.full(17, .3), None
        for bound in (0., .08, .15):
            with patch("probe_orca_reference_tube.actor_control", return_value=original):
                control = projected_actor(env, None, np.zeros((800, 17)), bound)
            obs, action, absolute, base, residual, desired, applied = control(
                None, jp.zeros(17), jp.zeros(17), jp.zeros(17), jp.zeros(17),
                jp.zeros((1, 3)), jp.zeros((1, 3)), {"step": jp.int32(0)})
            np.testing.assert_allclose(action, .3)
            np.testing.assert_allclose(obs, 1.)
            np.testing.assert_allclose(residual, bound)
            np.testing.assert_allclose(applied, min(bound, .1), atol=1e-7)
            self.assertLessEqual(float(jp.max(jp.abs(applied - base))), bound + 1e-6)

    def test_correction_metrics_are_not_pure_policy_placeholders(self):
        metrics = correction_metrics(dict(requested_residual=np.array([-.08, .02]),
                                          effective_residual=np.array([-.05, .01])))
        self.assertEqual(metrics["requested_residual_max_rad"], .08)
        self.assertEqual(metrics["effective_residual_max_rad"], .05)


if __name__ == "__main__":
    unittest.main()
