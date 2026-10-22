"""Exercise report aggregation on synthetic fixtures, never real experiment data."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report_orca_cylinder_study as report


class ReportTests(unittest.TestCase):
    def test_full_matrix_preserves_seed_units_and_controls(self):
        with tempfile.TemporaryDirectory(prefix="cylinder_report_test_") as directory:
            root = Path(directory)
            out = root / "study"
            out.mkdir()
            protocol = dict(train_jobs=[], eval_jobs=[], train_seeds=[41001, 41002, 41003],
                            budgets=[163840, 655360], conditions=["B0_nominal", "B1_position", "B5_push"])
            (out / "audit.json").write_text(json.dumps(dict(complete=True, test_episodes=4224)))

            def job(phase, method, seed, budget=0, condition="B0_nominal"):
                index = len(protocol[phase + "_jobs"])
                path = root / f"{phase}_{index}.json"
                run = root / f"{phase}_{index}"
                run.mkdir()
                path.write_text(json.dumps(dict(method=method, seed=seed, budget=budget,
                                                benchmark=condition, output=run.name)))
                protocol[phase + "_jobs"].append(path.name)
                return run

            for seed in protocol["train_seeds"]:
                for method in ("ppo", "guided", "residual"):
                    run = job("train", method, seed)
                    report.write_csv(run / "metrics.csv", [dict(step=step, clean_success30=.5,
                                     success30=.5, mean_hold=30., elapsed_sec=1.)
                                     for step in (0, 163840, 327680, 491520, 655360)])
            for condition in protocol["conditions"]:
                for method in ("ppo", "guided", "residual", "bc", "zero"):
                    seeds = protocol["train_seeds"] if method != "zero" else [41001]
                    budgets = protocol["budgets"] if method in ("ppo", "guided", "residual") else [0]
                    for seed in seeds:
                        for budget in budgets:
                            run = job("eval", method, seed, budget, condition)
                            folder = run / "step_000000000"
                            folder.mkdir()
                            count = dict(ppo=0, guided=16, residual=48, bc=32, zero=32)[method]
                            episodes = [dict(seed=110000+i, clean_success30=i<count, success30=i<count,
                                             success15=i<count, whole_episode_clean_success30=i<count,
                                             longest_hold_sec=30. if i<count else 0., drop=i>=count,
                                             substep_forbidden=False, post_release_penetration_m=.001,
                                             substep_force_peak_n=1., residual_rms_rad=.01)
                                        for i in range(64)]
                            (folder / "episodes.json").write_text(json.dumps(episodes))
            (out / "protocol.json").write_text(json.dumps(protocol))
            with patch.object(report, "ROOT", root), patch.object(report, "OUT", out):
                report.main()
            summary = json.loads((out / "report/summary.json").read_text())
            self.assertEqual(summary["test_episodes"], 4224)
            self.assertEqual(len(summary["groups"]), 24)
            for group in summary["groups"]:
                self.assertEqual(group["training_seeds"], 0 if group["method"] == "zero" else 3)
            with (out / "report/rl_contribution.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 36)
            for row in rows:
                self.assertEqual(float(row["difference"]), -.25 if row["method"] == "guided" else .25)
            pairs = json.loads((out / "report/paired_descriptive.json").read_text())
            self.assertTrue(all(p["paired_signflip_p"] == .25 for p in pairs))
            for name in ("learning_curves.png", "nominal_budgets.png", "robustness.png"):
                self.assertGreater((out / "report" / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
