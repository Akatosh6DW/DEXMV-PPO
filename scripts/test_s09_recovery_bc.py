import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from contextlib import redirect_stderr
import io
from unittest.mock import patch
import numpy as np
from run_s09_recovery_bc import mask_arrays, BCOptimizerProxy
import run_s09_recovery_bc
from run_s09_bc_phase_curriculum import KEEP
from run_s09_bc_conditioned import condition_arrays


class RecoveryBCTests(unittest.TestCase):
    def test_feedback_source_cannot_silently_become_phase_only(self):
        argv = ["run_s09_recovery_bc.py", "--bc-source", "unused.pkl", "--feedback-source", "--phase-only"]
        with patch("sys.argv", argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            run_s09_recovery_bc.main()
        self.assertEqual(error.exception.code, 2)

    def test_explicit_split_keeps_mode_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            obs = np.zeros((4,40,123), dtype=np.float32)
            obs[...,74:77] = [0,1,0]
            action = np.broadcast_to(np.arange(4)[:,None,None], (4,40,17)).copy()
            np.savez(folder/"data.npz", obs=obs, action=action, active=np.ones((4,40),bool))
            (folder/"dataset_manifest.json").write_text(json.dumps(dict(train_episodes=3)))
            arrays = condition_arrays(folder/"data.npz", np.r_[np.eye(3), -np.eye(3)])
            for i in (0,2,4):
                np.testing.assert_array_equal(np.asarray(arrays[i])[:,96], 1)
                np.testing.assert_array_equal(np.asarray(arrays[i])[:,95], 0)
            self.assertEqual(len(arrays[0]), 90)
            np.testing.assert_array_equal(np.asarray(arrays[5]), 3)

    def test_mask_only_observations_and_leave_actions_unchanged(self):
        arrays = tuple(np.ones((3,123)) if i%2==0 else np.ones((3,17)) for i in range(6))
        masked = mask_arrays(arrays, True)
        for i, value in enumerate(masked):
            if i%2==0:
                np.testing.assert_array_equal(np.asarray(value)[:,~KEEP], 0)
                np.testing.assert_array_equal(np.asarray(value)[:,KEEP], 1)
            else:
                self.assertIs(value, arrays[i])
        for before, after in zip(arrays, mask_arrays(arrays, False)):
            self.assertIs(before, after)

    def test_optimizer_override_is_local(self):
        module = SimpleNamespace(adam=lambda x:x, apply_updates=object())
        proxy = BCOptimizerProxy(module, .0001)
        self.assertEqual(proxy.adam(.001), .0001)
        self.assertEqual(module.adam(.001), .001)
        self.assertIs(proxy.apply_updates, module.apply_updates)


if __name__ == "__main__":
    unittest.main()
