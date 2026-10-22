"""Focused coverage for directory-only archival and portable legacy links."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import merge_reporting_materials as merge


class Tests(unittest.TestCase):
    def test_legacy_root_relative_link_maps_to_current_source(self):
        source = merge.LEGACY_PACKS[0] / "background.md"
        actual = merge.ROOT / "docs/experiments/S04/README.md"
        target = merge.OUT / "背景说明/background.md"
        mapped = merge.OUT / "S04_方块/文档/README.md"
        missing = []
        result = merge.rewrite("[S04](docs/experiments/S04/README.md)", source, target,
                               {actual: mapped}, {}, missing)
        self.assertEqual(result, "[S04](../S04_方块/文档/README.md)")
        self.assertFalse(missing)

    def test_directory_link_only_resolves_when_payload_will_exist(self):
        source = merge.ROOT / "docs/experiments/S04/README.md"
        data = merge.ROOT / "artifacts/orca_benchmark_suite/S04"
        missing = []
        result = merge.rewrite("[data](../../../artifacts/orca_benchmark_suite/S04/)", source,
            merge.OUT / "S04_方块/文档/README.md", {data / "protocol.json": merge.OUT / "S04_方块/数据/protocol.json"},
            {data: merge.OUT / "S04_方块/数据"}, missing)
        self.assertEqual(result, "[data](../数据)")
        self.assertFalse(missing)

    def test_external_link_untouched(self):
        link = "[paper](https://example.org/paper)"
        self.assertEqual(merge.rewrite(link, Path("/tmp/a"), Path("/tmp/b"), {}, {}, []), link)

    def test_weights_excluded(self):
        self.assertTrue({".pkl", ".pickle", ".pt", ".pth", ".ckpt"}.issubset(merge.EXCLUDED))
        self.assertFalse(merge.EXT.intersection(merge.EXCLUDED))

    def test_attachment_closure_excludes_weights_and_directory_contents(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            child = root / "child.md"
            image = root / "plot.png"
            source.write_text("[child](child.md) [weights](model.pkl) [folder](other/)")
            child.write_text("![plot](plot.png) [back](source.md)")
            image.write_bytes(b"test")
            (root / "model.pkl").write_bytes(b"weights")
            (root / "other").mkdir()
            (root / "other/unused.md").write_text("not linked individually")
            out = root / "delivery"
            mapping = {source: out / "source.md"}
            with patch.object(merge, "ROOT", root), patch.object(merge, "OUT", out):
                merge.supplement_links(mapping)
            self.assertEqual(set(mapping), {source, child, image})


if __name__ == "__main__":
    unittest.main()
