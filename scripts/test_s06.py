"""Smoke checks for cylinder geometry, initialization and substep audit state."""

from dataclasses import fields
from pathlib import Path
import unittest

import mujoco
import numpy as np

from diagnose_orca_grasp import model_indices
from orca_s06_env import CylinderPhysics, OrcaS06Grasp, cylinder_contacts
from orca_s05_env import PadData
from probe_hic53_orca import cube_penetration
from orca_cylinder_s02_env import cylinder_half_height

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/orca_benchmark_suite/S06/P02_aligned_wrist"


class CylinderTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(INPUT / "model.xml"))
        with np.load(INPUT / "reference.npz") as z:
            self.packet = dict(z)

    def test_real_cylinder_and_open_reset(self):
        m, p = self.model, self.packet
        self.assertEqual(m.geom("cube").type[0], mujoco.mjtGeom.mjGEOM_CYLINDER)
        np.testing.assert_allclose(m.geom("cube").size[:2], [.0265, .0265])
        self.assertAlmostEqual(m.body("cube").mass[0], .035)
        self.assertAlmostEqual(cylinder_half_height(p["object_quaternion"]), .0265)
        np.testing.assert_array_equal(p["initial_ctrl"][1:], np.zeros(16))
        d = mujoco.MjData(m)
        qids, oid, _ = model_indices(m)
        d.qpos[qids] = p["initial_ctrl"]
        d.qpos[oid:oid+7] = np.r_[p["object_position"], p["object_quaternion"]]
        mujoco.mj_forward(m, d)
        self.assertEqual(cube_penetration(m, d), 0)

    def test_audit_no_cross_episode_leak(self):
        m, p = self.model, self.packet
        fingers = ("thumb", "index", "middle", "ring", "pinky")
        mask = np.array([[(m.geom(i).name or "").startswith(f"orca_{f}_") for i in range(m.ngeom)] for f in fingers])
        forbidden = np.flatnonzero(~mask.any(0) & (np.arange(m.ngeom) != m.geom("cube").id))
        physics = CylinderPhysics(m, mask, "elliptic", 10, forbidden, True)
        d = mujoco.MjData(m)
        qids, oid, _ = model_indices(m)
        d.qpos[qids] = p["initial_ctrl"]
        d.qpos[oid:oid+7] = np.r_[p["object_position"], p["object_quaternion"]]
        d.mocap_pos[0] = p["object_position"]-[0, 0, .0305]
        state = physics.single("reset", d.qpos, d.qvel, p["initial_ctrl"], d.mocap_pos)
        args = (state.integration_words, p["ctrl"][0], d.mocap_pos, np.zeros(3))
        first = physics.single("forced_step", *args)
        physics.pad_audit[:] = 999
        second = physics.single("forced_step", *args)
        for field in fields(PadData):
            np.testing.assert_array_equal(getattr(first, field.name), getattr(second, field.name))

    def test_action_size_uses_native_model(self):
        env = object.__new__(OrcaS06Grasp)
        env._mj_model, env._mjx_model = self.model, None
        self.assertEqual(env.action_size, 17)

    def test_smaller_timestep_preserves_control_period(self):
        path = INPUT.parent / "P07_dt25"
        env = OrcaS06Grasp(path / "model.xml", "residual", path / "reference.npz")
        self.assertAlmostEqual(env.dt, .05)
        self.assertAlmostEqual(env.sim_dt, .0025)
        self.assertEqual(env.n_substeps, 20)
        self.assertEqual(env.native.substeps, 20)

    def test_trained_state_has_three_real_palmar_loads(self):
        path = INPUT.parent / "P09_implicit"
        model = mujoco.MjModel.from_xml_path(str(path / "model.xml"))
        data = mujoco.MjData(model)
        trace = INPUT.parent / "T08_residual_implicit/step_000020480/preview_trajectories.npz"
        with np.load(trace) as z:
            for field, key in (("qpos", "qpos"), ("qvel", "qvel"), ("ctrl", "ctrl"), ("mocap_pos", "mocap")):
                getattr(data, field)[:] = z[key][0, 99]
        mujoco.mj_forward(model, data)
        good, bad, _ = cylinder_contacts(model, data)
        self.assertTrue(np.all(good > .02))
        self.assertTrue(np.all(bad <= .02))

    def test_old_two_finger_case_is_not_three_finger_success(self):
        old = ROOT / "artifacts/orca_cylinder_s02"
        model = mujoco.MjModel.from_xml_path(str(old / "inputs/model.xml"))
        data = mujoco.MjData(model)
        with np.load(old / "eval/B0_nominal/residual_s1001/step_000000000/preview_trajectories.npz") as z:
            for field, key in (("qpos", "qpos"), ("qvel", "qvel"), ("ctrl", "ctrl"), ("mocap_pos", "mocap")):
                getattr(data, field)[:] = z[key][0, 99]
        mujoco.mj_forward(model, data)
        good, _, _ = cylinder_contacts(model, data)
        self.assertEqual(good[2], 0.)


if __name__ == "__main__":
    unittest.main()
