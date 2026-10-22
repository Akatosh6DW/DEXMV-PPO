"""Validate portable document links and weight exclusion before archival."""

from pathlib import Path
import csv
import tempfile
import unittest
from unittest.mock import patch

import package_s05_s08 as package


class Tests(unittest.TestCase):
    def test_refresh_preserves_locally_edited_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            payload=root/"report.md"
            payload.write_text("original")
            with (root/"文件来源与校验.csv").open("w") as f:
                writer=csv.DictWriter(f,fieldnames=["file","package_sha256"])
                writer.writeheader()
                writer.writerow(dict(file=payload.name,package_sha256=package.digest(payload)))
            with patch.object(package,"BUNDLE",root):
                package.check_refresh_targets()
                payload.write_text("reader edit")
                with self.assertRaisesRegex(ValueError,"changed locally"):
                    package.check_refresh_targets()
            self.assertEqual(payload.read_text(),"reader edit")

    def test_relative_link_follows_copied_file(self):
        source=Path("/tmp/project/docs/report.md")
        asset=Path("/tmp/project/data/result.csv")
        root=Path("/tmp/bundle")
        target=root/"S08/docs/report.md"
        mapping={asset:root/"S08/data/result.csv"}
        missing=[]
        with patch.object(package,"BUNDLE",root):
            result=package.rewrite_markdown("[data](../data/result.csv)",source,target,mapping,missing)
        self.assertEqual(result,"[data](../data/result.csv)")
        self.assertFalse(missing)

    def test_cross_study_absolute_link_is_portable(self):
        source=Path("/tmp/project/docs/report.md")
        asset=Path("/tmp/project/old/results.md")
        root=Path("/tmp/bundle")
        target=root/"S08/docs/report.md"
        with patch.object(package,"BUNDLE",root):
            result=package.rewrite_markdown(f"[old]({asset}#table)",source,target,
                {asset:root/"S07/docs/results.md"},[])
        self.assertEqual(result,"[old](../../S07/docs/results.md#table)")

    def test_web_and_page_anchors_unchanged(self):
        text="[web](https://example.org/a) [here](#section)"
        self.assertEqual(package.rewrite_markdown(text,Path("/tmp/a"),Path("/tmp/b"),{},[]),text)

    def test_missing_old_asset_not_silently_broken_link(self):
        root=Path("/tmp/bundle")
        missing=[]
        with patch.object(package,"BUNDLE",root):
            result=package.rewrite_markdown("[old](missing.csv)",Path("/tmp/project/report.md"),
                root/"report.md",{},missing)
        self.assertEqual(len(missing),1)
        self.assertNotIn("](missing.csv)",result)

    def test_no_pkl_or_duplicate_data_videos(self):
        files=package.inventory()
        self.assertTrue(files)
        self.assertFalse(any(p.suffix==".pkl" for p in files))
        self.assertFalse(any(p.suffix==".mp4" and p.is_relative_to(package.DATA) for p in files))
        self.assertTrue(all(p.is_relative_to(package.ROOT) for p in files))

    def test_demonstration_sources_included(self):
        files=package.inventory()
        for path in ("artifacts/orca_collision_refit/O01_hic53_source_object/source_correspondence.npz",
                     "artifacts/cube20/hic20/videos/20/input_rgb.mp4",
                     "artifacts/orca_cylinder_s02/source/selection96/selected_segment.npz"):
            self.assertIn(package.ROOT/path,files)


if __name__=="__main__":
    unittest.main()
