"""Reference resets preserve integrated physics, clear credit and stay training-only."""

import unittest
import jax
import jax.numpy as jp
import numpy as np

import run_s09_job as runner
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_reference_starts import ReferenceStarts
from collect_s09_reference_starts import clear_credit, eligible


class ReferenceStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = runner.ROOT/"artifacts/orca_benchmark_suite/S09"
        cls.env = ModeConditionedGrasp(root/"inputs/model.xml","pure")
        with np.load(root/"demonstrations/trajectories.npz") as z:
            commands = jp.asarray(z["action"][0,:40])
        def collect(key):
            initial = cls.env.reset(key)
            return jax.lax.scan(lambda s,a:(cls.env.step(s,a),None),initial,commands)[0]
        cls.state = jax.device_get(jax.jit(collect)(jax.random.PRNGKey(90504)))
        cls.bank = jax.tree_util.tree_map(lambda x:np.asarray(x)[None],clear_credit(cls.state))

    def test_zero_probability_keeps_original_reset(self):
        key = jax.random.PRNGKey(240101)
        expected = jax.jit(self.env.reset)(key)
        actual = jax.jit(ReferenceStarts(self.env,self.bank,0).reset)(key)
        self.assertEqual(int(actual.info["rsi_index"]),-1)
        actual = actual.replace(info={k:v for k,v in actual.info.items() if k!="rsi_index"})
        for a,b in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(actual)):
            np.testing.assert_array_equal(a,b)

    def test_demo_reset_preserves_dynamics_and_physics(self):
        wrapped = ReferenceStarts(self.env,self.bank,1)
        state = jax.jit(wrapped.reset)(jax.random.PRNGKey(2))
        self.assertEqual(int(state.info["step"]),40)
        self.assertEqual(int(state.info["hold_steps"]),0)
        np.testing.assert_array_equal(state.data.integration_words,self.bank.data.integration_words[0])
        action = self.env.ctrl_to_action(state.data.ctrl)
        expected = jax.jit(self.env.step)(state,action)
        actual = jax.jit(wrapped.step)(state,action)
        np.testing.assert_array_equal(expected.data.integration_words,actual.data.integration_words)
        self.assertEqual(float(actual.reward),float(expected.reward))

    def test_credit_and_phase_limit(self):
        state = self.state.replace(info=dict(self.state.info,hold_steps=np.int32(400),best_steps=np.int32(400)))
        cleared = clear_credit(state)
        self.assertEqual(int(cleared.info["hold_steps"]),0)
        self.assertEqual(int(cleared.info["best_steps"]),0)
        np.testing.assert_array_equal(cleared.data.integration_words,state.data.integration_words)
        wrapped = ReferenceStarts(self.env,self.bank,1)
        state = wrapped.reset(jax.random.PRNGKey(2))
        state = state.replace(info=dict(state.info,step=jp.int32(799)))
        actual = jax.jit(wrapped.step)(state,self.env.ctrl_to_action(state.data.ctrl))
        self.assertEqual(float(actual.done),1.)

    def test_reject_invalid_states(self):
        with self.assertRaises(ValueError):
            ReferenceStarts(self.env,self.bank,1.01)
        bad = self.state.replace(info=dict(self.state.info,max_depth=np.float32(.004)))
        self.assertFalse(eligible(bad,40))
        bad = self.state.replace(done=np.float32(1))
        self.assertFalse(eligible(bad,40))


if __name__ == "__main__":
    unittest.main()
