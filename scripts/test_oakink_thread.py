"""Passive mechanism and restored-state tests, not learned-task success tests."""

import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from oakink_cap_dynamics import apply_thread_contact_mode, step_passive
from probe_oakink_thread import probe

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/orca_formal/CAP21_thread_stop_and_filter"
REFERENCE = ROOT / "artifacts/orca_formal/CAP10_sequence_mount_ik/kinematic_reference.npz"


class OakInkThreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL / "model.xml").exists():
            raise unittest.SkipTest("Development cap model fixture unavailable")
        cls.meta = json.loads((MODEL / "report.json").read_text())

    def test_contact_filter_is_scoped_and_reversible(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL / "model.xml"))
        data = mujoco.MjData(model)
        self.assertEqual(model.nu, 17)
        self.assertFalse(any(model.joint(int(j)).name.startswith("cap_") for j in model.actuator_trnid[:, 0]))
        before = model.geom_conaffinity.copy()
        object_bodies = [model.body(n).id for n in ("oakink_cap", "oakink_bottle")]
        unaffected = ~np.isin(model.geom_bodyid, object_bodies)
        apply_thread_contact_mode(model, data)
        engaged = model.geom_conaffinity.copy()
        np.testing.assert_array_equal(engaged[unaffected], before[unaffected])
        self.assertTrue(model.jnt_limited[model.joint("cap_turn").id])
        data.eq_active[:] = False
        apply_thread_contact_mode(model, data)
        np.testing.assert_array_equal(model.geom_conaffinity[unaffected], before[unaffected])
        self.assertFalse(model.jnt_limited[model.joint("cap_turn").id])
        mujoco.mj_resetData(model, data)
        apply_thread_contact_mode(model, data)
        np.testing.assert_array_equal(model.geom_conaffinity, engaged)

    def test_isolated_torque_disengages_but_zero_does_not(self):
        with np.load(REFERENCE) as packet:
            reference = dict(packet)
        _, zero = probe(MODEL / "model.xml", self.meta, reference, "isolated_zero", seconds=2.)
        trace, positive = probe(MODEL / "model.xml", self.meta, reference, "isolated_torque", seconds=2.)
        self.assertFalse(zero["release_ever"])
        self.assertTrue(positive["release_ever"])
        self.assertLess(positive["max_engaged_thread_error_mm"], .1)
        self.assertEqual(float(trace["imposed_test_torque_nm"].max()), .005)
        self.assertFalse(positive["learned_policy"])

    def test_integration_restore_does_not_inherit_other_episode_contact_mode(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL / "model.xml"))
        data = mujoco.MjData(model)
        qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        data.qpos[qids], data.ctrl[:] = self.meta["initial_ctrl"], self.meta["initial_ctrl"]
        apply_thread_contact_mode(model, data)
        mujoco.mj_forward(model, data)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        release = self.meta["parameters"]["release_angle_rad"]
        step_passive(model, data, release)
        expected = np.empty_like(state)
        mujoco.mj_getState(model, data, expected, spec)
        data.eq_active[:] = False
        apply_thread_contact_mode(model, data)
        mujoco.mj_setState(model, data, state, spec)
        step_passive(model, data, release)
        actual = np.empty_like(state)
        mujoco.mj_getState(model, data, actual, spec)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
