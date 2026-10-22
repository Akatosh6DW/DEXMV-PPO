import unittest
from types import SimpleNamespace

import jax.numpy as jp
import jax
import numpy as np

from run_s09_angular_feedback import zero_unused_rows, AngularFeedbackGrasp, ModeConditionedGrasp, FEATURE
from run_s09_job import ROOT


class AngularFeedbackTests(unittest.TestCase):
    def test_angular_measurement_changes_only_reserved_slots(self):
        model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
        env = AngularFeedbackGrasp(model, "pure")
        state = jax.jit(env.reset)(jax.random.PRNGKey(90523))
        data = state.data.replace(qvel=state.data.qvel.at[env.dof+3:env.dof+6].set(jp.array([1., -2., 80.])))
        before = ModeConditionedGrasp._obs(env, data, state.info)
        after = env._obs(data, state.info)
        for key in ("state", "privileged_state"):
            expected = np.asarray(before[key]).copy()
            expected[101:104] = [.1, -.2, 5.]
            np.testing.assert_array_equal(after[key], expected)
        self.assertEqual(env._config.angular_velocity_observation, FEATURE)

    def test_only_reserved_first_layer_rows_change(self):
        params = (SimpleNamespace(mean=dict(state=np.zeros(123), privileged_state=np.zeros(140))),
            dict(params=dict(hidden_0=dict(kernel=np.ones((123,4)), bias=np.arange(4)))),
            dict(params=dict(hidden_0=dict(kernel=np.ones((140,4)), bias=np.arange(4)))))
        result = zero_unused_rows(params)
        for index, dimension in ((1,123),(2,140)):
            expected = np.ones((dimension,4))
            expected[101:104] = 0
            np.testing.assert_array_equal(result[index]["params"]["hidden_0"]["kernel"], expected)
            np.testing.assert_array_equal(params[index]["params"]["hidden_0"]["kernel"], 1)
            np.testing.assert_array_equal(result[index]["params"]["hidden_0"]["bias"], np.arange(4))

    def test_nonzero_normalization_mean_rejected(self):
        params = (SimpleNamespace(mean=dict(state=np.ones(123))), {}, {})
        with self.assertRaises(AssertionError):
            zero_unused_rows(params)


if __name__ == "__main__":
    unittest.main()
