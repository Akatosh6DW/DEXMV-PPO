"""Sensor semantics and native-force checks for paired tactile ablations."""

from pathlib import Path
import unittest
from unittest.mock import patch

import jax.numpy as jp
import mujoco
import numpy as np

from orca_matched_env import finger_normal_forces, touch_observation, FINGERS, OrcaMatchedGrasp
from orca_hic53_env import OrcaHic53Grasp


class TactileTests(unittest.TestCase):
    def test_reference_variants_preserve_reset_and_slew(self):
        from orca_matched_env import reference_variant
        initial = np.zeros(2)
        source = np.array([[.03, -.04], [.08, -.02], [.15, -.21], [.15, -.21]])
        np.testing.assert_array_equal(reference_variant(source, initial, "video"), source)
        np.testing.assert_array_equal(reference_variant(source, initial, "initial"), np.zeros_like(source))
        endpoint = reference_variant(source, initial, "endpoint")
        np.testing.assert_allclose(endpoint[-1], source[-1])
        self.assertLessEqual(np.abs(np.diff(np.vstack([initial, endpoint]), axis=0)).max(), .100001)
        self.assertFalse(np.array_equal(endpoint, source))
        with self.assertRaises(ValueError):
            reference_variant(source, initial, "unknown")
        with self.assertRaises(ValueError):
            OrcaMatchedGrasp(None, "pure", None, reference_mode="endpoint")

    def test_invalid_residual_bounds_are_rejected_before_loading_models(self):
        for value in (0., -.1, .16, np.nan, np.inf):
            with self.assertRaises(ValueError):
                OrcaMatchedGrasp(None, "residual", None, residual_bound=value)

    def test_height_shaping_is_bounded_and_contact_gated(self):
        from orca_matched_env import height_retention
        z = np.array([.21, .2, .195, .19])
        expected = np.exp(-np.array([0., 0., 1., 2.]))
        np.testing.assert_allclose(height_retention(z, .2, True, True, True, xp=np), expected)
        np.testing.assert_allclose(height_retention(jp.asarray(z), .2, True, True, True), expected, rtol=1e-5)
        for gates in ((False, True, True), (True, False, True), (True, True, False)):
            np.testing.assert_array_equal(height_retention(z, .2, *gates, xp=np), 0.)

    def test_contact_derived_hold_counter_is_not_actor_input(self):
        env = OrcaMatchedGrasp.__new__(OrcaMatchedGrasp)
        fake = dict(state=jp.arange(113, dtype=jp.float32), privileged_state=jp.arange(130, dtype=jp.float32))
        with patch.object(OrcaHic53Grasp, "_obs", return_value=fake):
            obs = env._obs(None, {"touch_obs": jp.zeros(10)})
        self.assertEqual(obs["state"].shape, (123,))
        self.assertEqual(obs["privileged_state"].shape, (140,))
        self.assertEqual(float(obs["state"][94]), 0.)
        self.assertEqual(float(obs["privileged_state"][94]), 94.)

    def test_critic_keeps_identical_sensor_access_in_both_conditions(self):
        env = OrcaMatchedGrasp.__new__(OrcaMatchedGrasp)
        def base_obs(*_):
            return dict(state=jp.zeros(113), privileged_state=jp.zeros(130))
        touch = jp.arange(10, dtype=jp.float32) / 10
        with patch.object(OrcaHic53Grasp, "_obs", side_effect=base_obs):
            off = env._obs(None, dict(touch_obs=jp.zeros(10), critic_touch_obs=touch))
            on = env._obs(None, dict(touch_obs=touch, critic_touch_obs=touch))
        np.testing.assert_array_equal(off["privileged_state"], on["privileged_state"])
        np.testing.assert_array_equal(off["state"][-10:], 0.)
        np.testing.assert_array_equal(on["state"][-10:], touch)

    def test_solver_normal_force_and_shared_finger_contacts(self):
        mask = np.zeros((5, 12), bool)
        mask[0, 1] = True
        mask[1, 2] = True
        geometries = np.array([[1, 10], [2, 10], [1, 2], [-1, -1]])
        force = np.r_[np.full(6, .1), np.ones(6), np.full(6, .5)]
        observed = finger_normal_forces(geometries, [0, 6, 12, -1], force, mask)
        np.testing.assert_allclose(observed, [3.6, 9., 0, 0, 0])
        np.testing.assert_array_equal(finger_normal_forces(geometries, [-1] * 4, force, mask), 0)
        np.testing.assert_array_equal(finger_normal_forces(geometries, [-1] * 4, [], mask), 0)

    def test_elliptic_force_is_not_sum_of_tangential_components(self):
        from orca_contact_forces import normal_loads
        force = np.array([2., -.4, .3, -.1, 3., .8, -.6, .2])
        expected = [2., 3., 0., 0.]
        np.testing.assert_allclose(normal_loads([0, 4, -1, 7], force, "elliptic"), expected)
        np.testing.assert_allclose(normal_loads([0, 4, -1, 7], force, "elliptic", xp=jp), expected)
        with self.assertRaises(ValueError):
            normal_loads([0], force, "unknown")

    def test_off_has_same_shape_without_force_leak(self):
        force = np.array([10., -1., .02, .01, 2.5])
        on = touch_observation(force, True)
        off = touch_observation(force, False)
        self.assertEqual(on.shape, off.shape)
        np.testing.assert_allclose(on, [1, 0, .004, .002, .5, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(off, np.zeros(10))

    def test_native_solver_agrees_with_mj_contact_force(self):
        root = Path(__file__).resolve().parents[1] / "artifacts/orca_collision_refit/O15_hic53_residual_margin"
        if not (root / "native_seed0.npz").exists():
            self.skipTest("Matched physical trace not present")
        model = mujoco.MjModel.from_xml_path(str(root / "cube.xml"))
        data = mujoco.MjData(model)
        with np.load(root / "native_seed0.npz") as z:
            data.qpos[:] = z["qpos"][60]
            data.qvel[:] = z["qvel"][60]
            data.ctrl[:] = z["ctrl"][60]
            data.mocap_pos[:] = z["mocap"][60]
        membership = np.zeros((5, model.ngeom), bool)
        for i, finger in enumerate(FINGERS):
            for geom in range(model.ngeom):
                membership[i, geom] = model.geom(geom).name.startswith(f"orca_{finger}_")
        for cone in ("pyramidal", "elliptic"):
            model.opt.cone = 0 if cone == "pyramidal" else 1
            mujoco.mj_forward(model, data)
            predicted = finger_normal_forces(data.contact.geom, data.contact.efc_address,
                                              data.efc_force, membership, cone=cone)
            expected = np.zeros(5)
            for i, contact in enumerate(data.contact):
                value = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, value)
                expected += (membership[:, contact.geom1] | membership[:, contact.geom2]) * value[0]
            self.assertGreater(expected.sum(), .01)
            np.testing.assert_allclose(predicted, expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
