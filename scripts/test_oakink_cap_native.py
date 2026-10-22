"""Regression tests for explicit cap state and final-frame scoring."""

import json
import copy
from pathlib import Path
import unittest

import mujoco
import numpy as np

from oakink_cap_dynamics import apply_thread_contact_mode
from oakink_cap_native import CapNativePhysics
from train_oakink_cap import summarize
from audit_oakink_cap_run import audit_run, audit_trace
from oakink_cap_env import OakInkCap

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/orca_formal/CAP21_thread_stop_and_filter"


class CapNativeTests(unittest.TestCase):
    def setUp(self):
        if not (MODEL / "model.xml").exists():
            self.skipTest("Development cap fixture unavailable")
        self.model = mujoco.MjModel.from_xml_path(str(MODEL / "model.xml"))
        self.meta = json.loads((MODEL / "report.json").read_text())
        self.native = CapNativePhysics(self.model, self.meta["parameters"]["release_angle_rad"])
        self.ctrl = np.asarray(self.meta["initial_ctrl"])
        qpos = self.model.qpos0.copy()
        qpos[self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]] = self.ctrl
        self.initial = self.native.single("reset", qpos, self.ctrl)

    def test_release_in_other_call_does_not_contaminate_step(self):
        expected = self.native.single("step", self.initial.integration_words, self.ctrl)
        data = self.native.restore(self.initial.integration_words)
        data.eq_active[:] = False
        apply_thread_contact_mode(self.model, data)
        other = self.native.pack(data)
        self.native.single("step", other.integration_words, self.ctrl)
        actual = self.native.single("step", self.initial.integration_words, self.ctrl)
        np.testing.assert_array_equal(actual.integration_words, expected.integration_words)
        np.testing.assert_array_equal(actual.qpos, expected.qpos)
        self.assertTrue(actual.thread_active)

    def test_policy_cannot_inject_external_cap_torque(self):
        data = self.native.restore(self.initial.integration_words)
        data.xfrc_applied[self.native.cap, 5] = .005
        injected = self.native.pack(data)
        with self.assertRaisesRegex(ValueError, "forbids external"):
            self.native.single("step", injected.integration_words, self.ctrl)

    def test_batch_matches_independent_calls(self):
        controls = np.stack([self.ctrl, self.ctrl + .01])
        expected = [self.native.single("step", self.initial.integration_words, c) for c in controls]
        actual = self.native.batched("step", np.stack([self.initial.integration_words] * 2), controls)
        for i in range(2):
            np.testing.assert_array_equal(actual.integration_words[i], expected[i].integration_words)


class CapScoreTests(unittest.TestCase):
    def test_independent_audit_rejects_fake_success_and_excess_residual(self):
        directory = ROOT / "artifacts/orca_formal/CAP31_auditable_residual_smoke"
        if not directory.exists():
            self.skipTest("Auditable development rollout unavailable")
        self.assertTrue(audit_run(directory)["passed"])
        env = OakInkCap(MODEL, ROOT / "artifacts/orca_formal/CAP10_sequence_mount_ik/kinematic_reference.npz")
        with np.load(directory / "step_000004096/trajectories.npz") as z:
            trace = dict(z)
        changed = copy.deepcopy(trace)
        changed["success"][:, -1] = 1
        with self.assertRaises(AssertionError):
            audit_trace(env, changed)
        changed = copy.deepcopy(trace)
        changed["requested_residual"] += .2
        with self.assertRaises(AssertionError):
            audit_trace(env, changed)

    def test_past_success_and_post_terminal_release_are_not_final_success(self):
        trace = {key: np.zeros((2, 4)) for key in ("success", "released", "degrees", "hold_sec",
                  "hand_penetration", "forbidden", "reward")}
        trace["force"] = np.zeros((2, 4, 5))
        trace["active"] = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], bool)
        trace["alive"] = np.array([[1, 1, 1, 1], [1, 0, 0, 0]], bool)
        trace["success"][:] = [[0, 1, 1, 0], [0, 0, 1, 1]]
        trace["released"][1, 2:] = 1
        result = summarize(trace)
        self.assertEqual(result["success"], 0)
        self.assertEqual(result["release_rate"], 0)
        self.assertEqual(result["terminal_failure_rate"], .5)
        trace["success"][0, -1] = 1
        self.assertEqual(summarize(trace)["success"], .5)


if __name__ == "__main__":
    unittest.main()
