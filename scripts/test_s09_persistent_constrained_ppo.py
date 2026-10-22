"""The R2 composition must preserve optimizer metadata and restore patches."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import run_s09_persistent_constrained_ppo as probe


class PersistentOptimizerTests(unittest.TestCase):
    def test_metadata_describes_actual_reward_without_erasing_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original = dict(reward_version="S09-R0 unchanged", actor_learning_rate=3e-6,
                critic_learning_rate=3e-4, teacher_used_at_inference=False)
            for name in ("constraint_config.json", "improvement_config.json"):
                (output / name).write_text(json.dumps(original))
            probe.record_reward(output, 15.)
            for name in ("constraint_config.json", "improvement_config.json"):
                values = json.loads((output / name).read_text())
                self.assertEqual(values["reward_version"], "S09-R2_qualified_hold_progress")
                self.assertEqual(values["hold_progress_weight"], 15.)
                self.assertEqual(values["actor_learning_rate"], 3e-6)
                self.assertEqual(values["critic_learning_rate"], 3e-4)
                self.assertFalse(values["teacher_used_at_inference"])
                self.assertEqual(len(values["reward_source_sha256"]), 64)

    def test_patches_are_restored_even_on_failure(self):
        improvement = probe.constrained.improvement
        runner = improvement.runner
        original = (improvement.ModeConditionedGrasp, runner.OrcaS09Grasp, runner.train.train)

        def fail():
            self.assertIs(improvement.ModeConditionedGrasp.func, probe.persistent.PersistentConditionedGrasp)
            self.assertEqual(improvement.ModeConditionedGrasp.keywords["hold_weight"], 15.)
            raise RuntimeError("isolated failure")

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(sys, "argv",
                ["probe", "--method", "guided", "--output", directory]), \
                mock.patch.object(probe.constrained, "main", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "isolated failure"):
                probe.main()
        self.assertEqual((improvement.ModeConditionedGrasp, runner.OrcaS09Grasp, runner.train.train), original)


if __name__ == "__main__":
    unittest.main()
