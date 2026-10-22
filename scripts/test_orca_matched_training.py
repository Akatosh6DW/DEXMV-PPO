"""Matched-data masking, initialization and small BC optimization regressions."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp
import numpy as np

from train_orca_matched import dataset_arrays, fit_matched_bc, initialize_matched
from prepare_hic_orca_object import digest


class MatchedTrainingTests(unittest.TestCase):
    def test_data_split_masking_and_task_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "trajectories.npz"
            obs = np.zeros((4, 40, 123), dtype=np.float32)
            obs[:, :, -10:] = .5
            obs[:, :, 0] = np.arange(4)[:, None]
            active = np.ones((4, 40), bool)
            active[:, -1] = False
            actions = np.zeros((4, 40, 17), np.float32)
            np.savez_compressed(path, obs=obs, action=actions, active=active)
            manifest = dict(dataset_sha256=digest(path), model_sha256="model",
                            reference_sha256="reference", actor_dim=123)
            (root / "dataset_manifest.json").write_text(json.dumps(manifest))
            on = dataset_arrays(path, True, "model", "reference")
            off = dataset_arrays(path, False, "model", "reference")
            self.assertEqual(on[0].shape[0], 90)
            self.assertEqual(on[2].shape[0], 27)
            self.assertEqual(on[4].shape[0], 39)
            self.assertTrue(np.all(np.asarray(on[4])[:, 0] == 3))
            for i in (0, 2, 4):
                np.testing.assert_array_equal(off[i][:, :-10], on[i][:, :-10])
                np.testing.assert_array_equal(off[i][:, -10:], 0.)
            with self.assertRaises(ValueError):
                dataset_arrays(path, False, "other-model", "reference")
            with self.assertRaises(ValueError):
                dataset_arrays(path, False, "model", "reference", "native_mujoco_callback")

    def test_zero_residual_head_and_bc_updates(self):
        env = SimpleNamespace(arm="residual")
        net, params = initialize_matched(env, 201, .7)
        obs = jp.ones((16, 123), dtype=jp.float32)
        logits = net.policy_network.apply(params[0], params[1], {"state": obs})
        np.testing.assert_array_equal(logits[:, :17], 0.)
        target = jp.full((16, 17), .1)
        arrays = (obs, target, obs, target, obs, target)
        fitted, rows = fit_matched_bc(net, params, arrays, 20, 201)
        self.assertLess(rows[-1]["validation_loss"], rows[0]["validation_loss"])
        head_before = params[1]["params"]["hidden_2"]
        head_after = fitted[1]["params"]["hidden_2"]
        np.testing.assert_array_equal(head_before["bias"][17:], head_after["bias"][17:])
        self.assertTrue(all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(fitted)))


if __name__ == "__main__":
    unittest.main()
