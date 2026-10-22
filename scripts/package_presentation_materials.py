"""Create a verified reporting snapshot without moving live experiment inputs."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ["artifacts/orca_benchmark_suite/" + s for s in ("S02", "S03", "S04")]
STUDIES.append("artifacts/orca_cylinder_s02")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    plan = {}

    def add(path):
        original = path
        path = path.resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Not an available workspace file: " + str(original))
        rel = original.relative_to(ROOT)
        if rel in plan and plan[rel] != path:
            raise ValueError("Conflicting destination: " + str(rel))
        plan[rel] = path

    def tree(rel):
        for path in sorted((ROOT / rel).rglob("*")):
            if path.is_file() and not {"__pycache__", ".git"}.intersection(path.parts):
                add(path)

    for rel in STUDIES + [
        "docs/literature", "docs/experiments/S02", "docs/experiments/S03",
        "docs/experiments/S04", "docs/experiments/Orca_Video_PPO",
        "docs/experiments/S02_S04_comparison", "artifacts/orca_benchmark_suite/S02_S04_comparison",
        "scripts", "artifacts/cube20/hic20",
        "artifacts/orca_cube_acquisition/D01_quality_demos",
        "artifacts/orca_cube_acquisition/E01_flex50_refinement",
        "artifacts/orca_collision_refit/O01_hic53_source_object",
        "artifacts/orca_collision_refit/O20_hic53_elliptic",
        "deliverables/Orca_Video_PPO_S04/scene",
        "deliverables/Orca_Video_PPO_S04/reference",
    ]:
        tree(rel)
    for path in (ROOT / "docs").glob("*.md"):
        add(path)
    for name in ("cylinder_s02_execution.md", "cylinder_s02_results_explained.md"):
        add(ROOT / "docs/experiments" / name)
    for name in ("view.py", "RELEASE_NOTES.md"):
        add(ROOT / "Aero-Hand" / name)

    # Preserve every declared frozen dependency; report historic mismatches, never edit them.
    frozen = []
    for rel in STUDIES:
        for protocol in sorted((ROOT / rel).glob("*protocol.json")):
            data = json.loads(protocol.read_text())
            for name, expected in data.get("source_hashes", {}).items():
                source = ROOT / name
                add(source)
                actual = digest(source)
                frozen.append(dict(protocol=str(protocol.relative_to(ROOT)), file=name,
                                   expected=expected, actual=actual, matches=actual == expected))

    pending = [p for p in plan.values() if p.suffix in (".xml", ".mjcf")]
    seen = set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        xml = ET.parse(source).getroot()
        if xml.tag != "mujoco":
            continue
        compiler = xml.find("compiler")
        for node in xml.iter():
            if "file" not in node.attrib:
                continue
            path = Path(node.get("file"))
            if not path.is_absolute():
                key = {"mesh": "meshdir", "texture": "texturedir"}.get(node.tag)
                directory = compiler.get(key, compiler.get("assetdir", "")) if compiler is not None and key else ""
                path = source.parent / directory / path
            path = path.resolve()
            add(path)
            if node.tag == "include":
                pending.append(path)

    total = sum(p.stat().st_size for p in plan.values())
    if shutil.disk_usage(ROOT).free < total + 1024**3:
        raise OSError("Not enough space for snapshot plus reserve")
    out.mkdir(parents=True)
    records = []
    print(json.dumps(dict(stage="copy", files=len(plan), bytes=total)), flush=True)
    for i, (rel, source) in enumerate(sorted(plan.items()), 1):
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        sha = digest(source)
        shutil.copy2(source, target)
        if digest(target) != sha:
            raise ValueError("Copy mismatch: " + str(rel))
        records.append(dict(source=str(source.relative_to(ROOT)), path=str(rel),
                            bytes=target.stat().st_size, sha256=sha))
        if i % 3000 == 0:
            print(json.dumps(dict(copied=i, total=len(plan))), flush=True)

    indexes = out / "indexes"
    indexes.mkdir()
    for name, suffixes in (("weights", {".pkl"}), ("datasets", {".npz", ".csv"}),
                           ("figures_videos", {".mp4", ".png", ".pdf"})):
        with (indexes / (name + ".csv")).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
            writer.writeheader()
            for item in records:
                path = Path(item["path"])
                if path.suffix in suffixes and (name != "weights" or "params" in path.name):
                    writer.writerow({k: item[k] for k in writer.fieldnames})

    # A separate movable scene; original XML and its hashes stay untouched.
    portable = out / "portable_scenes/cylinder"
    portable.mkdir(parents=True)
    source = ROOT / "artifacts/orca_cylinder_s02/inputs/model.xml"
    xml = ET.parse(source)
    for node in xml.getroot().iter():
        if "file" in node.attrib:
            path = Path(node.get("file"))
            if not path.is_absolute():
                path = source.parent / path
            target = out / path.resolve().relative_to(ROOT)
            if not target.is_file():
                raise FileNotFoundError(target)
            node.set("file", os.path.relpath(target, portable))
    xml.write(portable / "model.xml", encoding="utf-8", xml_declaration=True)

    counts = {}
    for rel in STUDIES:
        with (out / rel / "report/episodes.csv").open() as stream:
            counts[rel] = sum(1 for _ in csv.DictReader(stream))
    expected = {STUDIES[0]: 77600, STUDIES[2]: 4100, STUDIES[3]: 77600}
    if any(counts[k] != v for k, v in expected.items()):
        raise ValueError("Unexpected complete-study episode counts")
    audit = dict(copy_verified=True, files=len(records), bytes=total, episode_rows=counts,
                 source_hash_entries=len(frozen), source_hash_mismatches=[r for r in frozen if not r["matches"]],
                 originals_moved=False, original_results_modified=False,
                 incomplete_studies=[STUDIES[1]], excludes="Conda envs, .git, unrelated old checkpoints, redundant archive zips")
    (out / "archive_manifest.json").write_text(json.dumps(dict(files=records, frozen=frozen), indent=2) + "\n")
    (out / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
