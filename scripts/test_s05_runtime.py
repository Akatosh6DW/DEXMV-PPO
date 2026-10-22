"""Replay identity and old-side-grip rejection using actual MuJoCo models."""

from dataclasses import fields
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import mujoco
import numpy as np

from orca_s05_env import PadPhysics, PadData
from s05_pad_contacts import pad_contacts


class RuntimeTests(unittest.TestCase):
    def test_retrained_grip_uses_both_distal_pads(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/orca_benchmark_suite/S05/P20_impratio_narrow/impratio_2.2/model.xml"))
        data = mujoco.MjData(model)
        path = ROOT / "artifacts/orca_benchmark_suite/S05/T08_pad_residual_long/step_000163840/preview_trajectories.npz"
        with np.load(path) as trace:
            data.qpos[:], data.qvel[:], data.ctrl[:] = [trace[k][0, 99] for k in ("qpos", "qvel", "ctrl")]
            data.mocap_pos[:] = trace["mocap"][0, 99]
        mujoco.mj_forward(model, data)
        good, bad, _ = pad_contacts(model, data)
        self.assertTrue(np.all(good > .02))
        self.assertTrue(np.all(bad <= .02))

    def test_actual_old_side_grip_is_not_new_success(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"))
        data = mujoco.MjData(model)
        path = ROOT / "artifacts/orca_benchmark_suite/S02/eval/B0_nominal/residual_s1001/step_000000000/preview_trajectories.npz"
        with np.load(path) as trace:
            data.qpos[:], data.qvel[:], data.ctrl[:] = [trace[k][0, 99] for k in ("qpos", "qvel", "ctrl")]
            data.mocap_pos[:] = trace["mocap"][0, 99]
        mujoco.mj_forward(model, data)
        good, bad, _ = pad_contacts(model, data)
        self.assertEqual(good[1], 0.)
        self.assertGreater(bad[1], .02)

    def test_callback_has_no_previous_episode_audit_leak(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/orca_benchmark_suite/S05/P09_long_refinement/model.xml"))
        fingers = ("thumb", "index", "middle", "ring", "pinky")
        membership = np.array([[model.geom(i).name.startswith(f"orca_{finger}_")
                               for i in range(model.ngeom)] for finger in fingers])
        forbidden = np.flatnonzero(~membership.any(0) & (np.arange(model.ngeom) != model.geom("cube").id))
        physics = PadPhysics(model, membership, "elliptic", 10, forbidden, True)
        with np.load(ROOT / "artifacts/orca_benchmark_suite/S05/P09_long_refinement/reference.npz") as reference:
            ctrl = reference["initial_ctrl"]
            data = mujoco.MjData(model)
            data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]] = ctrl
            qid = model.joint("cube_freejoint").qposadr[0]
            data.qpos[qid:qid + 7] = np.r_[reference["object_position"], reference["object_quaternion"]]
            data.mocap_pos[0] = reference["object_position"] - [0, 0, .0305]
        initial = physics.single("reset", data.qpos, data.qvel, ctrl, data.mocap_pos)
        args = (initial.integration_words, ctrl, data.mocap_pos, np.zeros(3))
        first = physics.single("forced_step", *args)
        physics.pad_audit[:] = 999
        second = physics.single("forced_step", *args)
        for field in fields(PadData):
            np.testing.assert_array_equal(getattr(first, field.name), getattr(second, field.name))
        self.assertTrue(np.isfinite(second.pad_audit).all())


if __name__ == "__main__":
    unittest.main()
