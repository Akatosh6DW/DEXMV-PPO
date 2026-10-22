"""Regression checks for HIC transforms, bridge provenance and hold counting."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_hic_cube_demo import convert, read_motion


ROOT = Path(__file__).resolve().parents[1]


class HicPipelineTests(unittest.TestCase):
    def test_motion_rejects_shear_even_with_unit_determinant(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.MOTION"
            path.write_text("bone\n1 1 0 0 1 0 0 0 1 0 0 0\n")
            with self.assertRaises(ValueError):
                read_motion(path)

    def test_source_frame_alignment_and_rigid_skinning(self):
        source = ROOT / "artifacts/cube20/hic20/models_motion/20"
        data = convert(source, 30)
        raw = read_motion(source / "Model_Object_Cube.MOTION")["Bone001"]
        expected = raw[1:] @ np.linalg.inv(raw[0])
        np.testing.assert_allclose(data["object_poses"], expected, atol=1e-7)
        self.assertEqual(data["hand_joints"].shape, (186, 21, 3))
        self.assertEqual(data["source_motion_frame"][0], 1)
        self.assertEqual(data["source_video_frame"][0], 0)
        self.assertAlmostEqual(data["time"][-1], 185 / 30)

    def test_rejects_invalid_timebase(self):
        for fps in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                convert(Path("unused"), fps)

    def test_bridge_has_learned_provenance_and_real_timebase(self):
        d = np.load(ROOT / "artifacts/cube20/hic20/ppo_bridge/demo_dataset.npz")
        meta = json.loads(d["metadata_json"].item())
        self.assertIn("learned_prior.npz", meta["source"])
        self.assertIn("BC", meta["bc_report"]["algorithm"])
        self.assertEqual(d["aero_prior_ctrl"].shape, (64, 6))
        np.testing.assert_allclose(np.diff(d["time"]), .05)
        self.assertLessEqual(d["time"][-1], meta["stop_source_time_sec"])
        self.assertTrue(np.isfinite(d["aero_prior_ctrl"]).all())

    def test_independent_metrics_match_saved_frames(self):
        folder = ROOT / "artifacts/cube20/hic20/eval_H01_initial"
        d = np.load(folder / "trajectories.npz")
        report = json.loads((folder / "report.json").read_text())
        for i, row in enumerate(report["episodes"]):
            mask = d["strict_hold"][i]
            edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
            longest = np.max(edges[1::2] - edges[::2], initial=0) * report["dt"]
            self.assertAlmostEqual(longest, row["continuous_unsupported_sec"])
            self.assertTrue(np.all(d["alive"][i][mask]))
            self.assertFalse(np.any(d["support_contact"][i][mask]))
            self.assertFalse(np.any(d["palm_contact"][i][mask]))

    def test_final_bundle_hashes_and_continuous_hold(self):
        folder = ROOT / "artifacts/cube20/final_hic_bc_residual_ppo_5pct"
        manifest = json.loads((folder / "manifest.json").read_text())
        for name, expected in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), expected)
        config = json.loads((folder / "env_config.json").read_text())
        self.assertEqual(config["residual_prior"]["blend_alpha"], .05)
        self.assertEqual(Path(config["residual_prior"]["dataset_path"]), folder / "demo_dataset.npz")
        report = json.loads((folder / "report.json").read_text())
        d = np.load(folder / "trajectories.npz")
        self.assertEqual(len(report["episodes"]), 16)
        for i, row in enumerate(report["episodes"]):
            mask = d["strict_hold"][i]
            edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
            duration = np.max(edges[1::2] - edges[::2], initial=0) * report["dt"]
            self.assertAlmostEqual(duration, row["continuous_unsupported_sec"])
            self.assertGreater(duration, 20)
            self.assertTrue(np.all(d["alive"][i]))
            self.assertTrue(np.all(d["primary_count"][i][mask] >= 2))
            self.assertFalse(np.any(d["support_contact"][i][mask]))
            self.assertFalse(np.any(d["palm_contact"][i][mask]))


if __name__ == "__main__":
    unittest.main()
