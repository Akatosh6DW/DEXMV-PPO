"""Fast publication integrity and restoration safety tests (standard library)."""

import json
from pathlib import Path
import tempfile
import unittest

from restore_developer_release import ROOT, contained, digest, restore


class ReleaseTests(unittest.TestCase):
    def test_reject_parent_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("../other", "/etc/passwd"):
                with self.assertRaises(ValueError):
                    contained(Path(tmp), name)

    def test_reject_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "out").symlink_to("/etc", target_is_directory=True)
            with self.assertRaises(ValueError):
                contained(Path(tmp), "out/passwd")

    def test_s12_manifest_and_bytes(self):
        report = restore(verify_only=True)
        self.assertGreater(report["verified_unique_payloads"], 100)
        manifest = json.loads((ROOT / "release/developer_manifest.json").read_text())
        policies = [k for k in manifest["files"] if k.startswith(
            "artifacts/orca_benchmark_suite/S12/study_v2/training/") and
            len(Path(k).parts) == 7 and k.endswith("/params.pkl")]
        self.assertEqual(len(policies), 9)
        traces = [k for k in manifest["files"] if k.startswith(
            "artifacts/orca_benchmark_suite/S12/study_v2/evaluation/") and k.endswith(".npz")]
        self.assertEqual(len(traces), 1560)

    def test_no_duplicate_regular_payloads(self):
        legacy = json.loads((ROOT / "release/manifest.json").read_text())
        manifest = json.loads((ROOT / "release/developer_manifest.json").read_text())
        paths = {ROOT / "release" / p for p in legacy["files"]}
        paths.update(ROOT / "release" / e["storage"] for e in manifest["files"].values() if "storage" in e)
        hashes = {}
        for path in sorted(paths):
            if path.is_symlink():
                continue
            key = digest(path)
            self.assertNotIn(key, hashes, f"Repeated payload: {path} and {hashes.get(key)}")
            hashes[key] = path


if __name__ == "__main__":
    unittest.main()
