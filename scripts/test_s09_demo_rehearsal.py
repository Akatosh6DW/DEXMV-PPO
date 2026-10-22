"""Rehearsal must use the same conditioned input and train-only episodes."""

import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_demo_rehearsal import load_rehearsal
from orca_s09_env import cylinder_axis


class RehearsalTests(unittest.TestCase):
    def test_real_dataset_split_modes_and_unmasked_feedback(self):
        root = Path(__file__).resolve().parents[1] / "artifacts/orca_benchmark_suite/S09"
        path = root / "demonstrations/trajectories.npz"
        arrays = load_rehearsal(path, root / "inputs")
        self.assertEqual(len(arrays), 4)
        manifest = json.loads((path.parent / "dataset_manifest.json").read_text())
        with np.load(path, allow_pickle=False) as archive:
            obs, actions, active = archive["obs"].copy(), archive["action"], archive["active"].astype(bool)
        with np.load(root / "inputs/reference.npz", allow_pickle=False) as archive:
            axes = np.asarray(jax.vmap(cylinder_axis)(jp.asarray(archive["object_quaternion"])))
        mode = np.argmax(np.abs(obs[:, 0, 74:77] @ axes.T), axis=1)
        obs[..., 95:101] = np.eye(6, dtype=np.float32)[mode, None, :]
        split = manifest.get("train_episodes", len(obs) * 3 // 4)
        training = np.arange(len(obs))[:, None] < split
        early = np.arange(obs.shape[1])[None, :] < 30
        for i, mask in enumerate((active & training & early, active & training & ~early)):
            np.testing.assert_array_equal(arrays[2*i], obs[mask])
            np.testing.assert_array_equal(arrays[2*i+1], actions[mask])
        self.assertEqual(split, 42)
        self.assertTrue(np.any(np.asarray(arrays[0])[:, :93] != 0))


if __name__ == "__main__":
    unittest.main()
