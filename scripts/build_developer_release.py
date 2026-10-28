"""Extend the historical release with deduplicated training and audit assets.

Maintainer-only: run in the original experiment workspace. Runtime users use
restore_developer_release.py instead. Existing experiment files are never edited.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
TAG = "developer-2026-10-28"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    args = parser.parse_args()
    args.archives.mkdir(parents=True, exist_ok=True)
    legacy = json.loads((RELEASE / "manifest.json").read_text())
    canonical = {}
    dedup_count = dedup_bytes = 0
    # Keep the old manifest and paths valid; duplicate payloads become relative
    # links to an immutable release file, never to editable working source code.
    for relative, info in sorted(legacy["files"].items()):
        path = RELEASE / relative
        key = digest(path)
        if key != info["sha256"]:
            raise ValueError("Historical release changed: " + relative)
        if key in canonical:
            if not path.is_symlink():
                path.unlink()
                path.symlink_to(os.path.relpath(RELEASE / canonical[key], path.parent))
                dedup_count += 1
                dedup_bytes += info["bytes"]
        else:
            canonical[key] = str(path.resolve().relative_to(RELEASE))
    entries, remote, selected, excluded = {}, {}, {}, []

    def select(path, role="record", remote_only=False):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        rel = str(path.relative_to(ROOT))
        selected[rel] = (path, role, remote_only)

    for group in ("policies", "inputs"):
        for key, info in legacy["inventory"][group].items():
            for relative in info["paths"]:
                entries[relative] = dict(sha256=key, bytes=info["bytes"],
                    storage=canonical[key], role=group)
    for relative, info in legacy["files"].items():
        if relative.startswith("records/"):
            entries[relative[len("records/"):]] = dict(**info,
                storage=canonical[info["sha256"]], role="historical_record")

    # Training inputs were deliberately absent from the earlier replay release.
    datasets = {ROOT / "artifacts/orca_cube_acquisition/D01_quality_demos/trajectories.npz",
                ROOT / "artifacts/orca_cylinder_s02/inputs/demos/trajectories.npz"}
    for record in legacy["records"].values():
        config = json.loads((RELEASE / record["config"]).read_text())
        if config.get("dataset"):
            path = Path(config["dataset"])
            datasets.add(path if path.is_absolute() else ROOT / path)
    for dataset in sorted(datasets):
        select(dataset, "training_dataset")
        for path in dataset.parent.iterdir():
            if path.is_file() and path.suffix in (".json", ".csv"):
                select(path, "dataset_provenance")

    s12 = ROOT / "artifacts/orca_benchmark_suite/S12"
    study = s12 / "study_v2"
    protocol = json.loads((study / "protocol.json").read_text())
    for relative, expected in protocol["inputs"].items():
        path = ROOT / relative
        if digest(path) != expected:
            raise ValueError("Frozen S12 input changed: " + relative)
        if not relative.startswith("scripts/"):
            select(path, "s12_input")
    skip = {"package_before_plot_finalization", "interrupted_20261023_1800",
            "evaluation_interrupted_20261024", "video_renders_before_snapshot_fix", "__pycache__"}
    for folder in [study, *sorted(s12.glob("V2_*"))]:
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or any(p in skip for p in path.parts):
                continue
            if path.suffix not in (".json", ".jsonl", ".csv", ".md", ".py", ".xml", ".pkl", ".npz", ".png", ".mp4", ".log"):
                continue
            role = "s12_final" if folder == study else "s12_development"
            is_trace = (path.suffix == ".npz" and path.name != "trajectories.npz"
                        and path != s12 / "V2_P08_raise_hand3mm/trajectory.npz")
            select(path, role, is_trace)
    # Exactly one curated video source, not the duplicate all-in-one report.
    for folder in (ROOT / "deliverables/S12_软体圆柱_手心视频_2026-10-24",
                   ROOT / "deliverables/抓取展示视频_2026-10-12",
                   ROOT / "artifacts/orca_benchmark_suite/S10/videos"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in (".mp4", ".md", ".json"):
                select(path, "curated_video")

    # Gather source metadata for the canonical S09/S10 reference and selection.
    for folder in (ROOT / "artifacts/orca_benchmark_suite/S09/inputs",
                   ROOT / "artifacts/orca_benchmark_suite/S10/inputs"):
        for path in sorted(folder.glob("*.json")):
            select(path, "reference_provenance")

    for relative, (path, role, is_trace) in sorted(selected.items()):
        key, size = digest(path), path.stat().st_size
        entry = dict(sha256=key, bytes=size, role=role)
        if key in canonical:
            entry["storage"] = canonical[key]
        elif is_trace:
            remote.setdefault(key, (path, size))
        else:
            target = (RELEASE / "records" / relative if path.suffix in
                      (".md", ".csv", ".json", ".jsonl", ".png", ".log") else
                      RELEASE / "blobs" / (key + path.suffix))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != key:
                raise ValueError("Conflicting release file: " + str(target))
            if not target.exists():
                shutil.copyfile(path, target)
            canonical[key] = str(target.relative_to(RELEASE))
            entry["storage"] = canonical[key]
        entries[relative] = entry

    # Store each trace once in bounded, deterministic, uncompressed tar shards;
    # NPZ payloads are already compressed. No tar extraction of arbitrary paths.
    archives, remote_index = {}, {}
    batch, size, number = [], 0, 0

    def flush():
        nonlocal batch, size, number
        if not batch:
            return
        name = f"trajectories-{number:03d}.tar"
        target = args.archives / name
        with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as archive:
            for key, path, length in batch:
                member = key + ".npz"
                info = tarfile.TarInfo(member)
                info.size, info.mode, info.mtime = length, 0o644, 0
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
                remote_index[key] = dict(archive=name, member=member)
        archives[name] = dict(sha256=digest(target), bytes=target.stat().st_size,
            url=f"https://github.com/Akatosh6DW/DEXMV-PPO/releases/download/{TAG}/{name}")
        batch, size, number = [], 0, number+1

    for key, (path, length) in sorted(remote.items()):
        if key in canonical:
            continue
        if size + length > 300_000_000:
            flush()
        batch.append((key, path, length))
        size += length
    flush()
    for entry in entries.values():
        if "storage" not in entry:
            key = entry["sha256"]
            if key in canonical:
                entry["storage"] = canonical[key]
            else:
                entry.update(remote_index[key])

    # Mesh-to-content mapping supports relocating historical absolute XML paths.
    assets = {}
    for key, scene in legacy["portable_scenes"].items():
        original = ET.parse(RELEASE / legacy["blobs"][key])
        portable_path = RELEASE / scene["file"]
        portable = ET.parse(portable_path)
        old = [n for n in original.getroot().iter() if "file" in n.attrib]
        new = [n for n in portable.getroot().iter() if "file" in n.attrib]
        if len(old) != len(new):
            raise ValueError("Scene asset count differs")
        for a, b in zip(old, new):
            asset = (portable_path.parent / b.get("file")).resolve()
            assets[a.get("file")] = dict(storage=str(asset.relative_to(RELEASE)), sha256=digest(asset))

    summary = dict(paths=len(entries), unique_payloads=len({e["sha256"] for e in entries.values()}),
        git_payload_bytes=sum((RELEASE / v).stat().st_size for v in set(
            e["storage"] for e in entries.values() if "storage" in e)),
        remote_unique_traces=len(remote_index), remote_bytes=sum(a["bytes"] for a in archives.values()),
        legacy_duplicates_replaced=dedup_count, legacy_duplicate_bytes=dedup_bytes)
    manifest = dict(schema=1, tag=TAG, original_root=str(ROOT), files=entries,
        archives=archives, xml_assets=assets, summary=summary,
        exclusions=["Original human videos and externally licensed raw corpora",
            "Duplicate deliverable bundles and abandoned/interrupted checkpoint directories",
            "S12 nonconforming V1 prototype traces (failure described in development log)",
            "Virtual environments, caches, credentials and machine-specific editor configuration"],
        resume="weights-only warm start; optimizer/RNG not available in historical checkpoints")
    (RELEASE / "developer_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    policy_rows, seen = [], set()
    for item in legacy["inventory"]["evaluations"]:
        key = item["policy_sha256"]
        if not key or (item["group"], item["method"], key) in seen:
            continue
        seen.add((item["group"], item["method"], key))
        original = legacy["inventory"]["policies"][key]["paths"][0]
        folder = (ROOT / original).parent
        if not (folder / "run_config.json").is_file():
            folder = folder.parent
        config = json.loads((folder / "run_config.json").read_text())
        policy_rows.append(dict(experiment=item["group"], method=item["method"],
            seed=config.get("seed"), checkpoint=original, sha256=key, storage=canonical[key],
            config=entries[str((folder / "run_config.json").relative_to(ROOT))]["storage"],
            role="actually_evaluated_historical_policy"))
    for seed in (25211,25212,25213):
        for method in ("ppo", "guided", "residual"):
            folder = f"artifacts/orca_benchmark_suite/S12/study_v2/training/{method}_s{seed}"
            for name in (("params.pkl", "bc_params.pkl") if method == "guided" else ("params.pkl",)):
                entry = entries[folder + "/" + name]
                policy_rows.append(dict(experiment="S12", method="bc_initial" if name.startswith("bc_") else method,
                    seed=seed, checkpoint=folder+"/"+name, sha256=entry["sha256"], storage=entry["storage"],
                    config=entries[folder+"/run_config.json"]["storage"],
                    role="BC_initial" if name.startswith("bc_") else "final_51200_steps"))
    with (RELEASE / "POLICIES.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(policy_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(policy_rows)
    lines = ["# Published Grasp Videos", "", "One link per unique payload. Original descriptive names are preserved below.", ""]
    video_keys = set()
    for original, entry in sorted(entries.items(), key=lambda item: (item[1]["role"] != "curated_video", item[0])):
        if original.endswith(".mp4") and entry["sha256"] not in video_keys:
            video_keys.add(entry["sha256"])
            lines.append(f"- [{Path(original).name}]({entry['storage']})")
    (RELEASE / "VIDEOS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
