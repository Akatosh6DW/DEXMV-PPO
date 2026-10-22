"""Coverage and snapshot checks for the matched teacher timing experiment."""

import unittest

import jax
import numpy as np

from probe_s09_recovery_timing import (
    ROOT, training_cases, fresh_prefix, verify_replay,
)
from run_s09_bc_conditioned import ModeConditionedGrasp
from s09_suspension import SuspendedCylinder


class RecoveryTimingTests(unittest.TestCase):
    def test_selection_preserves_failed_teachers(self):
        rows = [dict(seed=90523, search_requested=True, prefix_alive=True, impulse="none",
                     selected_offset=dict(recovery_candidate=False)),
                dict(seed=90531, search_requested=False)]
        self.assertEqual(training_cases(rows, [90523, 90531]), rows[:1])

    def test_selection_rejects_heldout_duplicates_and_impulses(self):
        row = dict(seed=90523, search_requested=True, prefix_alive=True, impulse="none")
        for rows, allowed in (([row], []), ([row, row], [90523]),
                              ([dict(row, impulse="x_pos")], [90523])):
            with self.assertRaises(ValueError):
                training_cases(rows, allowed)

    def test_snapshots_and_original_endpoint(self):
        model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
        env = ModeConditionedGrasp(model, "pure")
        state = jax.device_get(jax.jit(env.reset)(jax.random.PRNGKey(90523)))
        targets = np.tile(np.asarray(state.data.ctrl), (800, 1))
        sim = SuspendedCylinder(model.read_text())
        for prefix in (40, 60):
            snapshot = fresh_prefix(sim, state, targets, prefix)
            duplicate = fresh_prefix(sim, state, targets, prefix)
            np.testing.assert_array_equal(snapshot[0], duplicate[0])
            self.assertAlmostEqual(sim.data.time, prefix*.05, places=10)
            report, trace = verify_replay(sim, snapshot, targets[prefix:prefix+2])
            self.assertEqual(report["requested_steps"], 2)
            self.assertAlmostEqual(trace["time"][-1], (prefix+2)*.05, places=10)
        with self.assertRaises(ValueError):
            fresh_prefix(sim, state, targets, 20)


if __name__ == "__main__":
    unittest.main()
