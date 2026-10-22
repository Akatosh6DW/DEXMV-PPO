"""Adversarial checks for held-out identities and final-state success scoring."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import mujoco

from audit_orca_formal_evaluation import audit_reset_geometry, check_report_identity, independent_success
from probe_oakink_orca_retarget import rigid_alignment


class EvaluationAuditTests(unittest.TestCase):
    def test_reset_audit_rejects_fake_penetration_and_support(self):
        directory = Path(__file__).resolve().parents[1] / "artifacts/orca_formal/EVAL_ALL_SCENARIOS_SMOKE/s301_guided_off__nominal"
        if not directory.exists():
            self.skipTest("Development reset fixture unavailable")
        report = json.loads((directory / "report.json").read_text())
        initial = np.array(json.loads((Path(report["run"]) / "env_config.json").read_text())["initial_ctrl"])
        model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
        with np.load(directory / "trajectories.npz") as z:
            trace = {k: z[k] for k in ("reset_qpos", "reset_mocap", "initial_penetration_m")}
        audit_reset_geometry(model, trace, initial)
        changed = copy.deepcopy(trace)
        changed["initial_penetration_m"] += .001
        with self.assertRaises(AssertionError):
            audit_reset_geometry(model, changed, initial)
        changed = copy.deepcopy(trace)
        changed["reset_mocap"][..., 2] += .001
        with self.assertRaises(AssertionError):
            audit_reset_geometry(model, changed, initial)

    def test_identity_rejects_scenario_and_zero_reference_swaps(self):
        root = Path("/tmp/formal_fixture")
        job = dict(run="s301_residual_on", zero="initial", scenario="nominal")
        report = dict(run=str(root / job["run"]), scenario="nominal", overrides={},
                      zero_correction_baseline=True, reference_mode="initial",
                      seeds=list(range(25000, 25064)), development_smoke=False)
        check_report_identity(report, job, root)
        for key, wrong in (("reference_mode", "video"), ("scenario", "mass15"),
                           ("zero_correction_baseline", False), ("development_smoke", True)):
            changed = dict(report, **{key: wrong})
            with self.assertRaises(ValueError):
                check_report_identity(changed, job, root)

    def test_success_requires_valid_final_and_valid_reset(self):
        shape = (3, 800)
        trace = dict(active=np.ones(shape, bool), alive=np.ones(shape, bool),
                     released=np.ones(shape, bool), thumb=np.ones(shape), contacts=np.full(shape, 2),
                     forbidden=np.zeros(shape), cube_z=np.ones(shape), spawn_z=np.ones(shape),
                     initial_penetration_m=np.zeros(shape),
                     hold=np.broadcast_to((np.arange(800) + 1) * .05, shape).copy())
        trace["forbidden"][1, -1] = 1
        trace["hold"][1, -1] = 0
        trace["initial_penetration_m"][2] = .004
        result = independent_success(trace)
        np.testing.assert_array_equal(result["raw"], [True, False, True])
        np.testing.assert_array_equal(result["strict"], [True, False, False])
        corrupt = copy.deepcopy(trace)
        corrupt["hold"][1, -1] = 40
        with self.assertRaises(AssertionError):
            independent_success(corrupt)

    def test_rigid_retarget_alignment_has_no_scale_or_reflection(self):
        source = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
        target = source @ rotation.T + [.1, .2, -.1]
        r, t = rigid_alignment(source, target)
        np.testing.assert_allclose(source @ r.T + t, target, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1.)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
