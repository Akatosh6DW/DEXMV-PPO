"""Verify portable report links, checksums, videos and complete experiment counts."""

import json
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

from export_s09_showcase import DELIVERY
from retarget_s09_reference import digest
from run_s09_study import OUT, ROOT, check_protocol


def main():
    check_protocol()
    r0 = json.loads((OUT / "report/matrix_verification.json").read_text())
    r1 = json.loads((OUT / "reward_iteration/report/verification.json").read_text())
    assert r0["status"] == "matrix_verified" and r0["episodes"] == 2560
    assert r1["status"] == "verified" and r1["episodes"] == 480
    tests = json.loads((OUT / "report/unit_tests.json").read_text())
    assert tests["passed"] and tests["tests"] == 12 and tests["errors"] == 0 and tests["failures"] == 0
    copied = json.loads((DELIVERY / "文件清单.json").read_text())
    for row in copied:
        assert digest(DELIVERY / row["file"]) == row["sha256"], row["file"]
        assert digest(ROOT / row["source"]) == row["source_sha256"], "Repackage changed source: " + row["source"]
    links = 0
    broken = []
    parser = MarkdownIt().enable("table")
    for file in DELIVERY.rglob("*.md"):
        for block in parser.parse(file.read_text()):
            for token in block.children or []:
                target = token.attrGet("href") if token.type == "link_open" else token.attrGet("src") if token.type == "image" else None
                if not target:
                    continue
                url = urlsplit(target)
                if url.scheme or not url.path:
                    continue
                destination = (file.parent / unquote(url.path)).resolve()
                links += 1
                if not destination.is_relative_to(DELIVERY) or not destination.exists():
                    broken.append(dict(file=str(file.relative_to(DELIVERY)), target=target))
    assert not broken, broken
    videos = json.loads((DELIVERY / "主要视频索引.json").read_text())
    assert len(videos) == 16 and len({(v["orientation"], v["method"]) for v in videos}) == 8
    for row in videos:
        file = DELIVERY / row["file"]
        assert row["decoded"] and digest(file) == row["sha256"]
        subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(file), "-f", "null", "-"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert not list(DELIVERY.rglob("*.pkl")), "Report delivery must not include weights"
    assert not any(p.is_symlink() for p in DELIVERY.rglob("*")), "Portable package should not depend on symlinks"
    for candidate in json.loads((OUT / "reward_iteration/report/candidate_policies.json").read_text()):
        assert digest(ROOT / candidate["params"]) == candidate["params_sha256"]
        assert digest(ROOT / candidate["model"]) == candidate["model_sha256"]
        if candidate["reference"]:
            assert digest(ROOT / candidate["reference"]) == candidate["reference_sha256"]
    target = DELIVERY / "交付核验.json"
    files = [dict(file=str(p.relative_to(DELIVERY)), bytes=p.stat().st_size, sha256=digest(p))
             for p in sorted(DELIVERY.rglob("*")) if p.is_file() and p != target]
    result = dict(status="verified", r0_episodes=2560, r1_episodes=480,
        copied_files=len(copied), total_files_excluding_this_report=len(files),
        bytes_excluding_this_report=sum(p["bytes"] for p in files), local_links_checked=links,
        broken_local_links=0, main_videos_decoded=16, total_videos=len(list(DELIVERY.rglob("*.mp4"))),
        copied_weight_files=0, symlinks=0, candidate_dependencies_verified=True, unit_tests_passed=12,
        scope="Portable documents/data/videos only; source excerpts and absolute original run configs are not a standalone training installation",
        files=files)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
