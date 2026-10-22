import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from probe_s09_recovery_student import KEEP, feedback_clamp, load_cases, digest


class RecoveryStudentTests(unittest.TestCase):
    def test_clamp_preserves_phase_and_mode_only(self):
        obs = np.arange(5*123, dtype=np.float32).reshape(5, 123)
        original = obs.copy()
        result = feedback_clamp(obs)
        np.testing.assert_array_equal(result[:, KEEP], obs[:, KEEP])
        np.testing.assert_array_equal(result[:, ~KEEP], np.broadcast_to(obs[0, ~KEEP], (5, (~KEEP).sum())))
        np.testing.assert_array_equal(obs, original)

    def test_ranked_requires_verified_complete_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/"report").mkdir()
            summary = folder/"summary.json"
            summary.write_text(json.dumps(dict(status="completed", smoke=False)))
            results = folder/"results.json"
            results.write_text(json.dumps([
                dict(seed=1, short_success=False, canonical=dict(qualified=True, trace_sha256="a")),
                dict(seed=2, short_success=True, canonical=dict(qualified=True, trace_sha256="b")),
                dict(seed=3, short_success=False, canonical=None)]))
            proof = dict(status="verified", results_sha256=digest(results),
                         summary_sha256=digest(summary), trace_hashes={"seed1/canonical_trace.npz": "a"})
            (folder/"report/evidence.json").write_text(json.dumps(proof))
            cases, source = load_cases(folder, True)
            self.assertEqual([r["seed"] for r in cases], [1])
            self.assertEqual(source, results)
            results.write_text("[]")
            with self.assertRaises(AssertionError):
                load_cases(folder, True)

    def test_legacy_collection_paths_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/"collection.json").write_text(json.dumps([
                dict(seed=5, qualified=True, trace_sha256="abc"), dict(seed=6, qualified=False)]))
            cases, _ = load_cases(folder, False)
            self.assertEqual(cases[0]["trace_file"], "seed5_canonical_trace.npz")
            self.assertEqual(len(cases), 1)


if __name__ == "__main__":
    unittest.main()
