"""Copy both complete experiment records, including S04's external training runs."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/orca_benchmark_suite"
DEFAULT = ROOT / "deliverables/Orca_S02_S04_Experiments"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Do not overwrite an existing experiment archive: " + str(out))
    plan = {}

    def add(source, dest):
        source = source.resolve()
        if not source.is_relative_to(ROOT):
            raise ValueError("External dependency needs explicit handling: " + str(source))
        if not source.is_file():
            raise FileNotFoundError(source)
        dest = Path(dest)
        if dest in plan and plan[dest] != source:
            raise ValueError("Archive path collision: " + str(dest))
        plan[dest] = source

    def tree(source, dest):
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path, Path(dest) / path.relative_to(source))

    def destination(source):
        rel = source.relative_to(ROOT)
        for name in ("S02", "S04"):
            prefix = Path("artifacts/orca_benchmark_suite") / name
            if rel.is_relative_to(prefix):
                return Path(name) / rel.relative_to(prefix)
        prefix = Path("artifacts/orca_benchmark_suite/S03/confirmation")
        if rel.is_relative_to(prefix):
            return Path("S04_training_sources") / rel.relative_to(prefix)
        return rel

    for name in ("S02", "S04"):
        tree(BASE / name, name)
        tree(ROOT / "docs/experiments" / name, "docs/experiments/" + name)
    for group in ("ppo_tuned", "guided_tuned", "residual_tuned"):
        tree(BASE / "S03/confirmation" / group, "S04_training_sources/" + group)
    for name in ("selected_recipes.json", "confirmation_protocol.json", "development_report.md",
                 "development_protocol.json", "training_audit.json"):
        source = BASE / "S03" / name
        if source.exists():
            add(source, source.relative_to(ROOT))
    frozen = {}
    for name in ("S02", "S04"):
        protocol = json.loads((BASE / name / "protocol.json").read_text())
        for rel, expected in protocol["source_hashes"].items():
            path = (ROOT / rel).resolve()
            if sha(path) != expected:
                raise ValueError("Frozen dependency changed: " + rel)
            add(path, destination(path))
            frozen[rel] = expected
    # Preserve the original repository layout for the reusable inputs and docs.
    for rel in (
        "artifacts/cube20/hic20", "artifacts/orca_cube_acquisition/D01_quality_demos",
        "artifacts/orca_cube_acquisition/E01_flex50_refinement",
        "artifacts/orca_collision_refit/O01_hic53_source_object",
        "artifacts/orca_collision_refit/O20_hic53_elliptic",
        "docs/experiments/Orca_Video_PPO", "docs/experiments/S02_S04_comparison",
        "artifacts/orca_benchmark_suite/S02_S04_comparison", "docs/experiments/S03",
    ):
        tree(ROOT / rel, rel)
    bundle = ROOT / "deliverables/Orca_Video_PPO_S04"
    tree(bundle / "scene", "shared/scene")
    tree(bundle / "reference", "shared/reference")
    for path in (bundle / "bridge/scripts").glob("*.py"):
        add(ROOT / "scripts" / path.name, "scripts/" + path.name)
    for name in ("analyze_orca_s02_s04.py", "package_orca_experiment_reading.py",
                 "package_orca_policy_bundle.py", "archive_orca_s02_s04.py"):
        add(ROOT / "scripts" / name, "scripts/" + name)
    # Gather original XML dependencies too; portable scenes live in shared/scene.
    pending = [p for p in plan.values() if p.suffix in (".xml", ".mjcf")]
    seen = set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        xml = ET.parse(source).getroot()
        compiler = xml.find("compiler")
        for node in xml.iter():
            if "file" not in node.attrib:
                continue
            path = Path(node.attrib["file"])
            if not path.is_absolute():
                key = "meshdir" if node.tag == "mesh" else "texturedir" if node.tag == "texture" else "assetdir"
                directory = compiler.get(key, compiler.get("assetdir", "")) if compiler is not None else ""
                path = source.parent / directory / path
            path = path.resolve()
            add(path, destination(path))
            if node.tag == "include":
                pending.append(path)
    required = sum(path.stat().st_size for path in plan.values())
    free = shutil.disk_usage(ROOT).free
    if free < required + 512 * 1024**2:
        raise OSError(f"Need {required} bytes plus 512 MiB reserve, have {free}")
    out.mkdir(parents=True)
    print(json.dumps(dict(stage="copy", files=len(plan), bytes=required), ensure_ascii=False), flush=True)
    records = []
    for i, (rel, source) in enumerate(sorted(plan.items())):
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(source)
        shutil.copy2(source, target)
        if sha(target) != expected:
            raise ValueError("Copy verification failed: " + str(rel))
        records.append(dict(source=str(source.relative_to(ROOT)), archived=str(rel),
                            bytes=target.stat().st_size, sha256=expected))
        if (i+1) % 2000 == 0:
            print(json.dumps(dict(copied=i+1, total=len(plan))), flush=True)
    # Internal aliases keep many original relative documentation links usable.
    aliases = {
        "artifacts/orca_benchmark_suite/S02": "S02",
        "artifacts/orca_benchmark_suite/S04": "S04",
        "artifacts/orca_benchmark_suite/S03/confirmation": "S04_training_sources",
        "S02/documents": "docs/experiments/S02",
        "S04/documents": "docs/experiments/S04",
    }
    for rel, target in aliases.items():
        link = out / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(out / target, link.parent), target_is_directory=True)
        if not link.resolve().is_relative_to(out):
            raise ValueError("Alias must remain inside archive")
    weight_index = []
    for run in sorted((out / "S02/train").iterdir()):
        if not run.is_dir():
            continue
        for name in ("params.pkl", "bc_params.pkl"):
            if (run / name).exists():
                weight_index.append(dict(experiment="S02", run=run.name,
                    kind="bc_initial" if name.startswith("bc_") else "final_163840",
                    file=str((run / name).relative_to(out))))
    for group in ("ppo_tuned", "guided_tuned", "residual_tuned"):
        for seed in range(21001, 21011):
            run = out / "S04_training_sources" / group / f"s{seed}"
            weight_index.append(dict(experiment="S04", run=f"{group}/s{seed}", kind="evaluated_163840",
                file=str((run / "step_000163840/params.pkl").relative_to(out))))
    for name in ("S02", "S04"):
        items = [r for r in weight_index if r["experiment"] == name]
        (out / name / "weights_index.json").write_text(json.dumps(items, indent=2)+"\n")
    counts = {}
    for name in ("S02", "S04"):
        with (out / name / "report/episodes.csv").open() as stream:
            counts[name] = sum(1 for _ in csv.DictReader(stream))
    if counts != {"S02": 77600, "S04": 4100}:
        raise ValueError("Episode populations differ")
    for rel, expected in frozen.items():
        if sha(out / destination(ROOT / rel)) != expected:
            raise ValueError("Missing archived frozen dependency: " + rel)
    audit = dict(complete=True, copied_files=len(records), copied_bytes=required,
                 episode_rows=counts, frozen_dependencies_verified=len(frozen),
                 aliases=aliases, originals_moved=False, new_training=False,
                 scope="Complete S02/S04 folders, three S04 source training groups, frozen dependencies and shared inputs",
                 not_claimed="Complete S03 experiment, standalone conda environment, arbitrary raw-video pipeline")
    (out / "archive_manifest.json").write_text(json.dumps(dict(audit=audit, files=records), indent=2)+"\n")
    (out / "verification.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
