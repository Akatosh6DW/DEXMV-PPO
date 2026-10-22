"""Record-only hold credit changes rewards, never the physical task."""

import unittest

import jax
import jax.numpy as jp
import numpy as np

from run_s09_record_hold_reward import RecordHoldGrasp, record_hold_delta
from run_s09_bc_conditioned import ModeConditionedGrasp
from s09_suspension import OUT


class RecordHoldTests(unittest.TestCase):
    def test_reward_is_subtraction_only(self):
        self.assertEqual(float(record_hold_delta(10, 11, True)), 0)
        self.assertEqual(float(record_hold_delta(10, 10, False)), 0)
        self.assertAlmostEqual(float(record_hold_delta(10, 10, True)), -.75)

    def test_main_credit_equals_longest_streak(self):
        for sequence, longest in (([1]*600, 600), ([1]*300+[0]+[1]*300, 300),
                                  ([1]*100+[0]+[1]*500, 500)):
            hold = best = 0
            reward = 0.
            for valid in sequence:
                hold = hold+1 if valid else 0
                next_best = max(best, hold)
                reward += .75*valid + float(record_hold_delta(best, next_best, valid))
                best = next_best
            self.assertAlmostEqual(reward, .75*longest)

    def test_physics_and_actor_unchanged(self):
        base = ModeConditionedGrasp(OUT/"inputs/model.xml", orientation_mode="web")
        revised = RecordHoldGrasp(OUT/"inputs/model.xml", orientation_mode="web")
        key = jax.random.PRNGKey(130001)
        a, b = base.reset(key), revised.reset(key)
        # Supply the same physically valid reference targets to both full-action
        # interfaces; this tests the reward, not the quality of a learned actor.
        with np.load(OUT/"inputs/reference.npz") as bank:
            controls, times = bank["ctrl"][int(a.info["reference_index"])], bank["time"]
        fa, fb = jax.jit(base.step), jax.jit(revised.step)
        changed = False
        for i in range(85):
            ctrl = np.array([np.interp((i+1)*.05, times, controls[:, j]) for j in range(17)])
            if i == 65:
                # A prior longer streak makes resumed valid frames lose main
                # credit. History is changed equally, never physical state.
                a = a.replace(info=dict(a.info, best_steps=jp.int32(600)))
                b = b.replace(info=dict(b.info, best_steps=jp.int32(600)))
            action = base.ctrl_to_action(jp.asarray(ctrl))
            a, b = fa(a, action), fb(b, action)
            np.testing.assert_array_equal(a.data.integration_words, b.data.integration_words)
            for name in a.obs:
                np.testing.assert_array_equal(a.obs[name], b.obs[name])
            for name in a.metrics:
                np.testing.assert_array_equal(a.metrics[name], b.metrics[name])
            self.assertEqual(float(a.done), float(b.done))
            self.assertAlmostEqual(float(b.reward-a.reward), float(b.metrics["record_hold_delta"]), places=6)
            changed |= float(b.metrics["record_hold_delta"]) < 0
        self.assertTrue(changed, "Must exercise actual reward subtraction")


if __name__ == "__main__":
    unittest.main()
