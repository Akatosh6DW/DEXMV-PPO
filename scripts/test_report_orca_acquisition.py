"""Synthetic fixtures for completeness checks and seed-level aggregation."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from report_orca_acquisition import main, cluster_interval, write_csv


class ReportTests(unittest.TestCase):
    def test_intervals_keep_seed_as_unit(self):
        self.assertEqual(cluster_interval([.5]), [None, None])
        self.assertEqual(cluster_interval([.5,.5,.5]), [.5,.5])
        low, high = cluster_interval([0,.5,1])
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 1)

    def fixture(self, root):
        jobs = []
        for jitter, scenario in ((.002,"nominal"),(.004,"offset4mm")):
            for method in ("ppo","guided","residual","bc","zero"):
                for seed in ([901] if method=="zero" else [901,902,903]):
                    name = f"{method}_s{seed}_{scenario}"
                    d = root / "eval" / name
                    (d / "step_000000000").mkdir(parents=True)
                    meta = dict(method="residual" if method=="zero" else method, seed=seed,
                                jitter=jitter, eval_count=2, eval_start=20000)
                    (d / "run_config.json").write_text(json.dumps(meta))
                    (d / "summary.json").write_text(json.dumps(dict(status="completed")))
                    success = method != "ppo"
                    rows = [dict(seed=s, success30=success, success15=success, longest_hold_sec=37 if success else 0,
                                 final_valid=success, finger_peak_n=10, forbidden=False,
                                 max_penetration_m=.001) for s in (20000,20001)]
                    (d / "step_000000000/episodes.json").write_text(json.dumps(rows))
                    jobs.append(dict(name="eval/"+name))
        for method in ("ppo","guided","residual","bc"):
            for seed in (901,902,903):
                d = root / "train" / f"{method}_s{seed}"
                d.mkdir(parents=True)
                (d / "run_config.json").write_text(json.dumps(dict(method=method,seed=seed)))
                rows = [dict(step=s, success30=0 if method=="ppo" else 1, mean_hold=20, mean_finger_peak_n=10)
                        for s in ([0] if method=="bc" else [0,100])]
                write_csv(d / "metrics.csv", rows)
        (root / "protocol.json").write_text(json.dumps(dict(jobs=jobs,total_test_episodes=52,main_ppo_steps=100)))

    def test_complete_aggregation_and_missing_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch.object(sys, "argv", ["report", "--root", str(root)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((root / "report/statistics.json").read_text())
            self.assertEqual(result["episodes"], 52)
            self.assertEqual(result["evaluations"], 26)
            for comparison in result["paired"]:
                difference = 1 if comparison["reference"]=="ppo" else 0
                self.assertEqual(comparison["success_difference"], difference)
                if comparison["reference_is_fixed_control"]:
                    self.assertIsNone(comparison["exact_seed_signflip_p"])
                else:
                    self.assertEqual(comparison["exact_seed_signflip_p"], .25 if difference else 1)
                self.assertEqual(comparison["paired_seed_bootstrap95"], [difference,difference])
            missing = root / "eval/zero_s901_nominal/summary.json"
            missing.unlink()
            with patch.object(sys, "argv", ["report", "--root", str(root)]):
                with self.assertRaisesRegex(ValueError, "Partial evaluation"):
                    main()


if __name__ == "__main__":
    unittest.main()
