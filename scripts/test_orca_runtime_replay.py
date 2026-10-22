"""The acceleration probe may not hide a changed initial physical trajectory."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from train_orca_matched import compare_initial_trace


class RuntimeReplayTests(unittest.TestCase):
    def test_exact_replay_and_metadata_validation(self):
        fields = ("method", "tactile", "seed", "eval_start", "eval_count", "model_sha256",
                  "reference_sha256", "actor_dim", "critic_dim", "init_std", "weights", "xla_flags")
        meta = dict.fromkeys(fields, "test")
        trace = dict(qpos=np.arange(12.).reshape(1, 3, 4), alive=np.ones((1, 3), dtype=bool))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "step_000000000").mkdir()
            (root / "run_config.json").write_text(json.dumps(meta))
            np.savez_compressed(root / "step_000000000/trajectories.npz", **trace)
            self.assertTrue(compare_initial_trace(root, trace, meta)["exact_match"])
            changed = dict(trace, qpos=trace["qpos"] + 1e-8)
            self.assertFalse(compare_initial_trace(root, changed, meta)["exact_match"])
            with self.assertRaises(ValueError):
                compare_initial_trace(root, trace, dict(meta, model_sha256="different"))
            with self.assertRaises(ValueError):
                compare_initial_trace(root, dict(trace, extra=np.zeros(1)), meta)


if __name__ == "__main__":
    unittest.main()
