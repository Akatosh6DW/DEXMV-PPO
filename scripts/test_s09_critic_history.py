"""History is privileged training information, not an actor behavior change."""

from pathlib import Path
import unittest

import run_s09_critic_history as history
from train_orca_matched import initialize_matched
import jax
import jax.numpy as jp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class CriticHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
        cls.old = history.persistence.persistent.PersistentConditionedGrasp(model)
        cls.new = history.RewardHistoryGrasp(model)

    def test_only_previously_zero_critic_slots_change(self):
        state = self.old.reset(jax.random.PRNGKey(42))
        info = dict(state.info, hold_steps=jp.int32(300))
        old, new = self.old._obs(state.data, info), self.new._obs(state.data, info)
        np.testing.assert_array_equal(old["state"], new["state"])
        indices = np.array(history.HISTORY_ROWS)
        np.testing.assert_array_equal(old["privileged_state"][indices], 0)
        np.testing.assert_array_equal(new["privileged_state"][indices], [1, .5])
        keep = np.ones(140, dtype=bool)
        keep[indices] = False
        np.testing.assert_array_equal(old["privileged_state"][keep], new["privileged_state"][keep])

    def test_features_use_current_history_not_future_success(self):
        info = dict(initial_penetration=jp.float32(0), max_depth=jp.float32(0),
                    ever_forbidden=jp.bool_(False), hold_steps=jp.int32(900))
        np.testing.assert_array_equal(jp.array(history.history_features(info)), [1, 1])
        for change in (dict(initial_penetration=jp.float32(.002)),
                       dict(max_depth=jp.float32(.004)), dict(ever_forbidden=jp.bool_(True))):
            self.assertEqual(float(history.history_features(dict(info, **change))[0]), 0.)

    def test_neutralization_changes_only_two_value_input_rows(self):
        net, before = initialize_matched(self.old, 41, .03)
        after = history.neutral_history_value(before)
        left = jax.tree_util.tree_flatten_with_path(before)[0]
        right = jax.tree_util.tree_flatten_with_path(after)[0]
        for (path, old), (_, new) in zip(left, right):
            keys = tuple(getattr(k, "key", getattr(k, "idx", None)) for k in path)
            if keys == (2, "params", "hidden_0", "kernel"):
                keep = np.ones(140, dtype=bool)
                keep[list(history.HISTORY_ROWS)] = False
                np.testing.assert_array_equal(np.asarray(old)[keep], np.asarray(new)[keep])
                np.testing.assert_array_equal(np.asarray(new)[list(history.HISTORY_ROWS)], 0)
            else:
                np.testing.assert_array_equal(old, new)
        state = self.old.reset(jax.random.PRNGKey(43))
        info = dict(state.info, hold_steps=jp.int32(300))
        old, new = self.old._obs(state.data, info), self.new._obs(state.data, info)
        a = net.value_network.apply(before[0], before[2], old)
        b = net.value_network.apply(after[0], after[2], new)
        np.testing.assert_array_equal(a, b)
        grads = jax.grad(lambda p: net.value_network.apply(after[0], p, new))(after[2])
        self.assertGreater(float(jp.abs(grads["params"]["hidden_0"]["kernel"][94]).sum()), 0)


if __name__ == "__main__":
    unittest.main()
