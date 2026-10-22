"""Focused regression checks for the fresh RGB-to-Orca experiment."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"Aero-Hand/sim_rl/mujoco_playground"))

import jax
import mujoco
import numpy as np

from orca_raw_can_bc import raw_dataset_arrays
from orca_raw_can_env import OrcaRawCanGrasp
from orca_residual_controls import compose
from run_orca_raw_can_job import fresh_path
import report_orca_raw_can_study as reporting

OUT = ROOT/"artifacts/orca_cylinder_raw_video_20261009"


class RawCanPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model, reference = OUT/"retarget_neutral/model.xml", OUT/"retarget_neutral/reference.npz"
        cls.pure = OrcaRawCanGrasp(model, "pure", reference)
        cls.residual = OrcaRawCanGrasp(model, "residual", reference)

    def test_old_data_path_rejected(self):
        with self.assertRaises(ValueError):
            fresh_path("artifacts/orca_cylinder_study/D04_reference/reference.npz")
        with self.assertRaises(ValueError):
            fresh_path("artifacts/orca_cylinder_raw_video_20261009/../orca_cylinder_study/reference.npz")

    def test_true_cylinder_and_neutral_start(self):
        geom = self.pure.mj_model.geom("cube")
        self.assertEqual(int(geom.type[0]), mujoco.mjtGeom.mjGEOM_CYLINDER)
        np.testing.assert_allclose(geom.size[:2], [.069644/2, .103637/2])
        np.testing.assert_array_equal(self.pure._config.initial_ctrl, np.zeros(17))
        self.assertIsNone(self.pure.mjx_model)

    def test_reference_not_exposed_to_pure_policy(self):
        np.testing.assert_array_equal(self.pure.reference_ctrl, 0.)
        state = jax.jit(self.pure.reset)(jax.random.PRNGKey(101))
        np.testing.assert_array_equal(state.obs["state"][95:112], 0.)
        self.assertEqual(state.obs["state"].shape, (123,))
        self.assertEqual(state.obs["privileged_state"].shape, (140,))
        self.assertEqual(float(state.obs["state"][94]), 0.)

    def test_paired_physical_initial_states(self):
        key = jax.random.PRNGKey(209001)
        a, b = (jax.jit(env.reset)(key) for env in (self.pure, self.residual))
        np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        np.testing.assert_array_equal(a.data.mocap_pos, b.data.mocap_pos)
        np.testing.assert_array_equal(a.data.ctrl, b.data.ctrl)

    def test_zero_residual_replays_reference(self):
        env = self.residual
        state = jax.jit(env.reset)(jax.random.PRNGKey(209002))
        step = jax.jit(env.step)
        for _ in range(3):
            state = step(state, np.zeros(17, np.float32))
            np.testing.assert_allclose(state.data.ctrl, state.info["video_base"], atol=1e-6)
            np.testing.assert_array_equal(state.info["requested_residual"], 0.)

    def test_requested_residual_is_bounded_nonintegrating(self):
        env = self.residual
        base = np.asarray(env.reference_ctrl[40])
        desired, residual = compose(base, np.full(17, 100.), .08, env._lowers, env._uppers)
        np.testing.assert_allclose(residual, .08)
        self.assertLessEqual(float(np.abs(desired-base).max()), .080001)
        again, _ = compose(base, np.full(17, 100.), .08, env._lowers, env._uppers)
        np.testing.assert_array_equal(desired, again)

    def test_early_failure_data_is_not_fabricated(self):
        path = OUT/"demonstrations/trajectories.npz"
        manifest = json.loads((path.parent/"dataset_manifest.json").read_text())
        a, b, c, d, val, val_action = raw_dataset_arrays(path, manifest["model_sha256"],
                                                       manifest["reference_sha256"], manifest["physics_backend"])
        with np.load(path) as data:
            split = len(data["obs"])*3//4
            np.testing.assert_array_equal(a, data["obs"][:split][data["active"][:split]])
            self.assertEqual(len(val), int(data["active"][split:].sum()))
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertEqual(len(val), len(val_action))

    def test_dataset_binding_rejects_wrong_reference(self):
        with self.assertRaises(ValueError):
            raw_dataset_arrays(OUT/"demonstrations/trajectories.npz", "wrong", "wrong", "wrong")

    def test_fresh_video_hash_chain(self):
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        hand = OUT/"perception/tomato41"
        obj = OUT/"perception/can41_v2"
        retarget = OUT/"retarget_neutral"
        h, o, r = [json.loads((p/"manifest.json").read_text()) for p in (hand, obj, retarget)]
        self.assertEqual(h["source_video_sha256"], digest(OUT/"source/tomato41_raw.mp4"))
        self.assertEqual(o["hand_sha256"], digest(hand/"hand_landmarks.npz"))
        self.assertEqual(r["source_packet_sha256"], digest(obj/"hand_object.npz"))
        self.assertEqual(r["reference_sha256"], digest(retarget/"reference.npz"))
        self.assertEqual(h["frames"], 102)

    def test_protocol_inputs_unchanged(self):
        protocol = json.loads((OUT/"protocol.json").read_text())
        for path, digest in protocol["frozen_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), digest, path)
        self.assertEqual(len(protocol["train_jobs"]), 9)
        self.assertEqual(len(protocol["eval_jobs"]), 66)

    def test_zero_success_does_not_imply_zero_uncertainty(self):
        low, high = reporting.interval(0, 64)
        self.assertEqual(low, 0.)
        self.assertAlmostEqual(high, 1-.025**(1/64))
        self.assertGreater(high, .05)
        self.assertLess(high, .06)

    def test_incomplete_matrix_cannot_publish_final_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"protocol.json").write_text(json.dumps(dict(frozen_sha256={})))
            with patch.object(reporting, "OUT", root), self.assertRaises(FileNotFoundError):
                reporting.main()
            self.assertFalse((root/"report").exists())


if __name__ == "__main__":
    unittest.main()
