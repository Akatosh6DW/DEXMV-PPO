"""Independent object compilation and causal replay path checks."""

import unittest

from run_orca_s03_job import MODEL, REFERENCE
from run_orca_suite_job import evaluator as original_evaluator
from orca_s03_env import OrcaS03Grasp
from orca_s03_objects import OrcaS03ObjectGrasp
from orca_s03_evaluation import evaluator, recorded_actions, assert_matched_prefix
from orca_hic53_env import task_packet
from train_orca_acquisition import initial_gate
from train_orca_matched import initialize_matched
import jax
import jax.numpy as jp
import mujoco
import numpy as np


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_pmap_shmap_merge", False)

    def test_object_models_are_explicit_transfers_not_source_bound_demos(self):
        for shape in ("cuboid", "sphere"):
            env = OrcaS03ObjectGrasp(MODEL, "residual", REFERENCE, shape=shape)
            self.assertTrue(initial_gate(env)["passed"])
            with self.assertRaises((AssertionError, ValueError)):
                task_packet(env.mj_model, REFERENCE)
            self.assertIs(env.native.model, env.mj_model)
            state = jax.jit(env.reset)(jax.random.PRNGKey(60050))
            self.assertAlmostEqual(float(state.info["cube_spawn_pos"][2] - state.data.mocap_pos[0, 2]),
                                   env.half_height + .004, places=6)
            nxt = jax.jit(env.step)(state, jp.zeros(17))
            self.assertTrue(np.isfinite(nxt.data.qpos).all())
            self.assertEqual(nxt.obs["state"].shape, (123,))
            if shape == "sphere":
                self.assertEqual(env.mj_model.geom("cube").type[0], mujoco.mjtGeom.mjGEOM_SPHERE)

    def test_recording_zeros_only_post_terminal_actions(self):
        trace = dict(active=np.array([[True, False]]), policy_action=np.ones((1, 2, 17)))
        actions = recorded_actions(trace)
        np.testing.assert_array_equal(actions[:, 0], 1)
        np.testing.assert_array_equal(actions[:, 1], 0)

    def test_replay_and_closed_loop_match_without_push_and_before_push(self):
        nominal = OrcaS03Grasp(MODEL, "residual", REFERENCE, audit=True)
        forced = OrcaS03Grasp(MODEL, "residual", REFERENCE, audit=True, push_force=.6)
        net, params = initialize_matched(nominal, 19002, .3)
        # Nonconstant state feedback is essential: zero residual alone is a weak test.
        head = params[1]["params"]["hidden_2"]
        head["kernel"] = head["kernel"].at[:, :17].set(.002)
        seeds = [60060, 60061]
        blank = np.zeros((2, 800, 17), np.float32)
        initial, closed = jax.device_get(evaluator(nominal, net, seeds)(params, blank))
        old_initial, old = jax.device_get(original_evaluator(nominal, net, seeds)(params))
        np.testing.assert_array_equal(initial, old_initial)
        for key in old:
            np.testing.assert_array_equal(old[key], closed[key], err_msg=key)
        actions = recorded_actions(closed)
        _, replay = jax.device_get(evaluator(nominal, net, seeds, replay=True)(params, actions))
        assert_matched_prefix(closed, replay)
        _, pushed_closed = jax.device_get(evaluator(forced, net, seeds)(params, blank))
        _, pushed_replay = jax.device_get(evaluator(forced, net, seeds, replay=True)(params, actions))
        assert_matched_prefix(closed, pushed_closed, length=200)
        assert_matched_prefix(closed, pushed_replay, length=200)
        np.testing.assert_array_equal(pushed_replay["policy_action"], actions)
        self.assertTrue(np.all(np.linalg.norm(pushed_closed["push"][:, 200:204], axis=-1) > .59))
        self.assertTrue(np.any(closed["alive"][:, 199]))


if __name__ == "__main__":
    unittest.main()
