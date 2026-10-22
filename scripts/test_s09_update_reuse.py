"""A data-reuse ablation must preserve all other training arguments."""

import unittest
from run_s09_update_reuse import update_options


class UpdateReuseTests(unittest.TestCase):
    def setUp(self):
        self.options = dict(num_envs=32, num_minibatches=4, unroll_length=20,
                            batch_size=32, num_updates_per_batch=4,
                            discounting=.99, learning_rate=3e-6, num_timesteps=40960)

    def test_only_reuse_changes(self):
        for passes in (1, 2, 4):
            changed = update_options(self.options, passes)
            self.assertEqual(changed.pop("num_updates_per_batch"), passes)
            expected = dict(self.options)
            expected.pop("num_updates_per_batch")
            self.assertEqual(changed, expected)
        self.assertEqual(self.options["num_updates_per_batch"], 4)

    def test_rejects_confounded_batch(self):
        for key, value in (("num_envs", 64), ("num_minibatches", 2),
                           ("unroll_length", 80), ("batch_size", 64),
                           ("num_updates_per_batch", 2)):
            with self.assertRaises(ValueError):
                update_options(dict(self.options, **{key:value}), 1)
        for invalid in (0, 3, 8):
            with self.assertRaises(ValueError):
                update_options(self.options, invalid)


if __name__ == "__main__":
    unittest.main()
