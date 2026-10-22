"""Trim only verified generated archive copies into a reporting-only collection."""

import csv
import json
from pathlib import Path

from package_presentation_materials import ROOT, STUDIES, digest

OUT = ROOT / "deliverables/汇报资料_2026-10-10"
TEXT = {".md", ".csv", ".json", ".jsonl", ".log", ".diff"}
FIGURES = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}


def keep(rel):
    p = Path(rel)
    if rel.startswith("docs/literature/"):
        return p.suffix in {".md", ".docx", ".pdf", ".tex", ".bib", ".bst", ".png"}
    if rel.startswith("docs/"):
        return p.suffix in TEXT | FIGURES
    if any(rel.startswith(s + "/") for s in STUDIES):
        if "frozen_sources" in p.parts:
            return False
        # Preserve numeric evidence and protocol metadata, not runnable snapshots.
        return p.suffix in TEXT | FIGURES
    if rel.startswith("artifacts/orca_benchmark_suite/S02_S04_comparison/"):
        return p.suffix in TEXT | FIGURES
    return rel.startswith("artifacts/") and p.suffix in TEXT | FIGURES


def main():
    manifest = json.loads((OUT / "archive_manifest.json").read_text())
    records = manifest["files"]
    retained = [r for r in records if keep(r["path"])]
    removed = [r for r in records if not keep(r["path"])]
    # Refuse to delete anything modified since this task created the archive.
    for record in records:
        rel = Path(record["path"])
        target = OUT / rel
        source = ROOT / record["source"]
        if rel.is_absolute() or ".." in rel.parts or target.is_symlink():
            raise ValueError("Unsafe archive path: " + str(rel))
        if not target.resolve().is_relative_to(OUT) or source.resolve().is_relative_to(OUT):
            raise ValueError("Not an independent archive copy: " + str(rel))
        if digest(target) != record["sha256"] or digest(source) != record["sha256"]:
            raise ValueError("Changed source/copy; refusing trim: " + str(rel))
    print(json.dumps(dict(validated=len(records), keep=len(retained), remove=len(removed))), flush=True)
    for record in removed:
        (OUT / record["path"]).unlink()
    # These generated indexes/scene were made by the same packaging task, not source inputs.
    for rel in ("indexes/weights.csv", "indexes/datasets.csv", "indexes/figures_videos.csv",
                "portable_scenes/cylinder/model.xml"):
        (OUT / rel).unlink(missing_ok=True)
    for directory in sorted(OUT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    indexes = OUT / "indexes"
    indexes.mkdir(exist_ok=True)
    for name, suffixes in (("data", {".csv", ".json", ".jsonl"}),
                           ("figures", FIGURES), ("documents", {".md", ".docx", ".pdf"})):
        with (indexes / (name + ".csv")).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
            writer.writeheader()
            writer.writerows({k: r[k] for k in writer.fieldnames} for r in retained
                            if Path(r["path"]).suffix in suffixes)
    counts = {}
    for study in STUDIES:
        with (OUT / study / "report/episodes.csv").open() as stream:
            counts[study] = sum(1 for _ in csv.DictReader(stream))
    assert counts == dict(zip(STUDIES, (77600, 91800, 4100, 77600)))
    forbidden = {".pkl", ".pickle", ".npz", ".npy", ".stl", ".msh", ".xml", ".zip", ".py"}
    assert not any(p.is_file() and p.suffix in forbidden for p in OUT.rglob("*"))
    audit = dict(scope="reporting data and explanations; not a runnable reproduction bundle",
                 files=len(retained), bytes=sum(r["bytes"] for r in retained),
                 removed_copy_bytes=sum(r["bytes"] for r in removed),
                 copy_sha256_verified=True, source_files_unchanged=True,
                 episode_rows=counts, incomplete_studies=[STUDIES[1]],
                 excluded=["weights", "models", "source code", "binary trajectories", "videos"],
                 note="Historical document links to excluded runtime artifacts may require the original project.")
    (OUT / "archive_manifest.json").write_text(json.dumps(dict(files=retained), indent=2) + "\n")
    (OUT / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
