"""Synthetic S03 comparison and report integration tests, not experiment data."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import report_orca_s03 as report
from report_orca_suite import holm


class StatisticsTests(unittest.TestCase):
    def test_paired_reference_and_primary_resolution(self):
        positive = report.contrast(np.ones((10, 200)), np.zeros((1, 200)), "positive", True)
        self.assertEqual(positive["difference_pp"], 100)
        self.assertAlmostEqual(positive["ci_low_pp"], 100)
        self.assertEqual(positive["paired_seed_p"], 2/1024)
        self.assertLess(holm([positive["paired_seed_p"]]*3)[0], .05)
        null = report.contrast(np.zeros((10, 200)), np.zeros((10, 200)), "same")
        self.assertEqual(null["paired_seed_p"], 1)
        with self.assertRaises(ValueError):
            report.contrast(np.zeros((1, 200)), np.zeros((1, 200)), "fake repetition")

    def test_report_reads_paired_modes_and_corrects_exactly_three_primary_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "artifacts/orca_benchmark_suite/S03"
            (out / "jobs").mkdir(parents=True)
            jobs = []
            for group in ("ppo_tuned", "guided_tuned", "residual_tuned", "mechanism"):
                for seed in range(21001, 21011):
                    name = f"synthetic/{group}/s{seed}"
                    paired = group == "mechanism"
                    job = dict(output="artifacts/orca_benchmark_suite/S03/"+name,
                               condition="mechanism" if paired else "B0_nominal",
                               group="residual_tuned" if paired else group,
                               method="residual" if paired else "ppo", seed=seed, budget=655360,
                               eval_start=71000, eval_count=2, paired_replay=paired)
                    jobs.append(job)
                    path = out / "jobs" / (name.replace("/", "__")+".json")
                    path.write_text(json.dumps(job))
                    directory = root / job["output"]
                    directory.mkdir(parents=True)
                    (directory / "summary.json").write_text(json.dumps(dict(status="completed", job_sha256=report.digest(path))))
                    modes = ("nominal_closed", "nominal_replay", "pushed_closed", "pushed_replay") if paired else ("evaluation",)
                    for mode in modes:
                        (directory / mode).mkdir()
                        good = group == "residual_tuned" or mode in ("nominal_closed", "nominal_replay", "pushed_closed")
                        episodes = [dict(seed=71000+i, success30=good, clean_success30=good,
                                         full_episode_clean_success30=good, longest_hold_sec=37. if good else 0.,
                                         substep_force_peak_n=1., max_penetration_m=.001,
                                         valid_before_push=True, valid_one_second_after_push=good) for i in range(2)]
                        (directory / mode / "episodes.json").write_text(json.dumps(episodes))
            protocol = dict(benchmark_names=["B0_nominal"], budgets=[163840, 655360],
                            ordinary_eval_count=30, paired_mechanism_jobs=10)
            (out / "confirmation_protocol.json").write_text(json.dumps(protocol))
            (out / "evaluation_protocol.json").write_text(json.dumps(dict(jobs=jobs,
                confirmation_protocol_sha256=report.digest(out / "confirmation_protocol.json"))))
            with patch.object(report, "ROOT", root), patch.object(report, "OUT", out), patch("sys.argv", ["report"]):
                report.main()
            statistics = json.loads((out / "report/statistics.json").read_text())
            self.assertTrue(statistics["complete"])
            self.assertEqual(statistics["episodes"], 140)
            primary = [c for c in statistics["comparisons"] if c["primary"]]
            self.assertEqual(len(primary), 3)
            self.assertTrue(all(c["supports_primary_positive_difference"] for c in primary))
            self.assertTrue(all(c["primary_holm_3_p"] == 6/1024 for c in primary))
            missing = root / jobs[0]["output"] / "summary.json"
            missing.unlink()
            with patch.object(report, "ROOT", root), patch.object(report, "OUT", out), patch("sys.argv", ["report"]):
                with self.assertRaises(RuntimeError):
                    report.main()


if __name__ == "__main__":
    unittest.main()
