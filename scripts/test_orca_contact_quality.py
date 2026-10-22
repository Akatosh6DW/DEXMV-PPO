"""Exercise contact-scope reporting through its public CLI with synthetic data."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("report_orca_contact_quality.py")


class ContactQualityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol = dict(train_seeds=[1001, 1002], test_seeds=list(range(40000, 40004)), jobs=[])
        self.rows = [
            dict(seed=40000, success30=True, clean_success30=True,
                 max_penetration_m=.003, post_release_penetration_m=.003),
            dict(seed=40001, success30=True, clean_success30=True,
                 max_penetration_m=.003001, post_release_penetration_m=.001),
            dict(seed=40002, success30=True, clean_success30=False,
                 max_penetration_m=.004, post_release_penetration_m=.004),
            dict(seed=40003, success30=False, clean_success30=False,
                 max_penetration_m=.001, post_release_penetration_m=.001),
        ]

    def job(self, label, completed=True, rows=None, summary_hash="fixture"):
        name = "eval/B0_nominal/" + label
        self.protocol["jobs"].append(dict(name=name, sha256="fixture"))
        if not completed:
            return
        directory = self.root / name
        (directory / "step_000000000").mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps(dict(
            status="completed", job_sha256=summary_hash)))
        (directory / "step_000000000/episodes.json").write_text(json.dumps(
            self.rows if rows is None else rows))

    def run_report(self, partial=False):
        (self.root / "protocol.json").write_text(json.dumps(self.protocol))
        return subprocess.run([sys.executable, str(SCRIPT), "--root", str(self.root),
                               *(["--allow-partial"] if partial else [])],
                              capture_output=True, text=True)

    def result(self):
        return json.loads((self.root / "report/contact_quality_supplement.json").read_text())

    def test_threshold_and_phase_scope(self):
        self.job("zero_video")
        process = self.run_report()
        self.assertEqual(process.returncode, 0, process.stderr)
        result = self.result()
        self.assertTrue(result["complete"])
        self.assertTrue(result["primary_metric_unchanged"])
        self.assertEqual(result["additional_hypothesis_tests"], 0)
        row = result["groups"][0]
        self.assertEqual(row["raw_success30"], .75)
        self.assertEqual(row["predefined_postrelease_clean"], .5)
        self.assertEqual(row["supplemental_all_phase_clean"], .25)
        self.assertEqual(row["primary_successes_failing_pre_release_depth"], 1)
        self.assertEqual(row["any_pre_only_penetration_failures"], 1)
        self.assertEqual(row["max_all_phase_penetration_mm"], 4)
        self.assertEqual(row["policies"], 1)

    def test_half_residual_groups_both_seeds(self):
        self.job("half_residual_s1001")
        self.job("half_residual_s1002")
        self.assertEqual(self.run_report().returncode, 0)
        row = self.result()["groups"][0]
        self.assertEqual(row["method"], "half_residual")
        self.assertEqual(row["policies"], 2)
        self.assertEqual(row["episodes"], 8)
        self.assertEqual(row["primary_successes_failing_pre_release_depth"], 2)

    def test_incomplete_group_is_not_a_complete_estimate(self):
        self.job("residual_s1001")
        self.job("residual_s1002", completed=False)
        self.assertEqual(self.run_report(partial=True).returncode, 0)
        result = self.result()
        self.assertFalse(result["complete"])
        self.assertEqual(result["groups"], [])
        self.assertEqual(len(result["missing"]), 1)
        self.assertNotEqual(self.run_report().returncode, 0)

    def test_rejects_summary_from_another_job(self):
        self.job("zero_video", summary_hash="wrong")
        self.assertNotEqual(self.run_report().returncode, 0)

    def test_rejects_wrong_test_case_order(self):
        self.job("zero_video", rows=self.rows[::-1])
        self.assertNotEqual(self.run_report().returncode, 0)

    def test_rejects_clean_success_without_raw_success(self):
        rows = [dict(row, success30=False, clean_success30=True) for row in self.rows]
        self.job("zero_video", rows=rows)
        self.assertNotEqual(self.run_report().returncode, 0)


if __name__ == "__main__":
    unittest.main()
