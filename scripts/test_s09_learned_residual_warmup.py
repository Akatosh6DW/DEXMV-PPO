"""The routed residual wrapper uses delayed Adam and restores local hooks."""

from pathlib import Path
import unittest

import run_s09_constrained_ppo as constrained
import run_s09_learned_residual_experts as routed
from run_s09_learned_residual_warmup import delayed_residual


class LearnedWarmupTests(unittest.TestCase):
    def test_context_restoration(self):
        split, train = constrained.split_adam, routed.learned.runner.train.train
        with self.assertRaisesRegex(RuntimeError,"probe"):
            with delayed_residual(2560,Path("/tmp/not-written")):
                self.assertIsNot(constrained.split_adam,split)
                self.assertIsNot(routed.learned.runner.train.train,train)
                raise RuntimeError("probe")
        self.assertIs(constrained.split_adam,split)
        self.assertIs(routed.learned.runner.train.train,train)

    def test_invalid_warmup(self):
        for value in (0, -2560, 100):
            with self.assertRaises(ValueError):
                with delayed_residual(value,Path("/tmp/not-written")):
                    pass


if __name__ == "__main__":
    unittest.main()
