"""Reject missing, duplicated, or stale whole-study evidence."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_orca_whole_study import ROOT, check_videos, collect_reports, contact_disposition, digest


class WholeStudyEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_missing_is_not_complete(self):
        self.write("shard/a/report.json", {})
        found, missing = collect_reports([self.root / "shard"], [{"label": "a"}, {"label": "b"}])
        self.assertEqual(set(found), {"a"})
        self.assertEqual(missing, ["b"])

    def test_duplicates_rejected_even_when_counts_match(self):
        for shard in ("one", "two"):
            self.write(shard + "/a/report.json", {})
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation"):
            collect_reports([self.root / "one", self.root / "two"], [{"label": "a"}, {"label": "b"}])

    def test_unexpected_development_result_rejected(self):
        self.write("shard/smoke/report.json", {})
        with self.assertRaisesRegex(ValueError, "Unexpected evaluation"):
            collect_reports([self.root / "shard"], [{"label": "formal"}])

    def test_clean_contact_does_not_require_negative_disposition(self):
        self.write("formal_result.json", {"penetration_max_mm": 1.5, "forbidden_fraction": 0})
        self.assertEqual(contact_disposition(self.root), "below_inspection_threshold")

    def make_review(self):
        result = self.write("formal_result.json", {"penetration_max_mm": 1.7, "forbidden_fraction": .051})
        inspection = self.write("contact_inspection.json", {"inspection": "test fixture"})
        return self.write("contact_review.json", dict(review_complete=True,
            disposition="retain_scored_negative_episodes_continue_unchanged_protocol",
            result_sha256=digest(result), inspection_sha256=digest(inspection),
            no_score_or_checkpoint_replacement=True, no_environment_reward_or_training_change=True))

    def test_review_retains_negative_result_but_cannot_follow_changed_evidence(self):
        self.make_review()
        self.assertEqual(contact_disposition(self.root), "retain_scored_negative_episodes_continue_unchanged_protocol")
        self.write("contact_inspection.json", {"inspection": "changed"})
        with self.assertRaisesRegex(ValueError, "Stale contact review"):
            contact_disposition(self.root)

    def test_contact_review_cannot_replace_bad_seed_or_checkpoint(self):
        path = self.make_review()
        review = json.loads(path.read_text())
        review["no_score_or_checkpoint_replacement"] = False
        path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError, "changed the formal experiment"):
            contact_disposition(self.root)

    def video_fixture(self):
        from render_orca_formal import select_cases
        model = self.write("model.xml", {"fixture": "model"})
        trace = self.write("trace.npz", {"fixture": "trace"})
        reports = {f"s{seed}_{method}_{touch}": dict(episodes=dict(success30=[False] * 64),
                   trace_sha256=digest(trace), model_xml_sha256=digest(model))
                   for seed in range(301, 306) for method in ("bc", "ppo", "guided", "residual")
                   for touch in ("off", "on")}
        cases = select_cases(reports)
        self.write("selection.json", cases)
        rows = []
        for index, case in enumerate(cases):
            rendering = dict(trace_sha256=digest(trace), trace=str(trace), source_model=str(model),
                renderer_source_sha256=digest(ROOT / "scripts/render_orca_modes.py"), episode=case["episode"],
                camera=dict(azimuth=225, elevation=-18, lookat=[.035, .025, .245], distance=.38),
                no_physics_resimulation=True, same_saved_states_both_modes=True, physics_arrays_match=True)
            self.write(f"video{index}/report.json", rendering)
            rows.append(dict(**case, success30=False, output=str(self.root / f"video{index}"),
                videos={mode: {"sha256": "stub", "ffprobe": {}} for mode in ("stl", "collision")}))
        self.write("manifest.json", dict(cases=rows, complete=True))
        return reports

    @patch("render_orca_formal.verify_video", return_value={"sha256": "stub", "ffprobe": {}})
    def test_all_failed_videos_still_require_complete_coverage(self, _):
        reports = self.video_fixture()
        result = check_videos(self.root, reports, "cube")
        self.assertTrue(result["passed"])
        self.assertEqual(result["cases"], 8)
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["cases"].pop()
        self.write("manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Missing required videos"):
            check_videos(self.root, reports, "cube")

    @patch("render_orca_formal.verify_video", return_value={"sha256": "stub", "ffprobe": {}})
    def test_video_wrong_camera_or_model_rejected(self, _):
        reports = self.video_fixture()
        rendering = json.loads((self.root / "video0/report.json").read_text())
        rendering["camera"]["azimuth"] = 45
        self.write("video0/report.json", rendering)
        with self.assertRaisesRegex(ValueError, "Wrong video camera"):
            check_videos(self.root, reports, "cube")
        rendering["camera"]["azimuth"] = 225
        self.write("video0/report.json", rendering)
        self.write("model.xml", {"fixture": "other model"})
        with self.assertRaisesRegex(ValueError, "Wrong rendered model"):
            check_videos(self.root, reports, "cube")


if __name__ == "__main__":
    unittest.main()
