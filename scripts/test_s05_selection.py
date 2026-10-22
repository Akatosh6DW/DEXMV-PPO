"""Guard against selecting an untrained prior or using test data for selection."""

import json
from pathlib import Path
import tempfile
import unittest

from evaluate_s05_selected import select, wilson


class SelectionTests(unittest.TestCase):
    def test_zero_step_excluded_and_latest_tie_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [dict(step=s, success30=1., pad_hold_sec=37.05) for s in (0, 40960, 81920)]
            (root / "summary.json").write_text(json.dumps(dict(status="completed", rows=rows)))
            for row in rows:
                path = root / f"step_{row['step']:09d}"
                path.mkdir()
                (path / "params.pkl").write_bytes(str(row["step"]).encode())
            self.assertEqual(select(root)["step"], 81920)

    def test_success_rate_precedes_duration_and_recency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [dict(step=10, success30=1., pad_hold_sec=35.),
                    dict(step=20, success30=.75, pad_hold_sec=36.)]
            (root / "summary.json").write_text(json.dumps(dict(status="completed", rows=rows)))
            path = root / "step_000000010"
            path.mkdir()
            (path / "params.pkl").write_bytes(b"fixture")
            self.assertEqual(select(root)["step"], 10)

    def test_small_perfect_sample_is_not_certain_population_success(self):
        lower, upper = wilson(32, 32)
        self.assertAlmostEqual(lower, .89282, places=4)
        self.assertAlmostEqual(upper, 1.)


if __name__ == "__main__":
    unittest.main()
