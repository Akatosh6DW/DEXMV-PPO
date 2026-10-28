"""Verify and restore release inputs without overwriting local experiment work."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, name):
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Unsafe destination: " + name)
    return path


def relocated_xml(source, manifest, release):
    root = ET.fromstring(source.read_bytes())
    changed = False
    for node in root.iter():
        name = node.get("file")
        if name is None:
            continue
        asset = manifest["xml_assets"].get(name)
        if asset is None:
            raise ValueError("Unresolved XML asset: " + name)
        path = contained(release, asset["storage"])
        if digest(path) != asset["sha256"]:
            raise ValueError("Corrupt XML asset: " + name)
        node.set("file", str(path.resolve()))
        changed = True
    return ET.tostring(root, encoding="utf-8") if changed else source.read_bytes()


def restore(root=ROOT, *, all_records=False, trajectories=False, archive_dir=None, verify_only=False):
    release = root / "release"
    manifest = json.loads((release / "developer_manifest.json").read_text())
    verified, restored, existing, conflicts, relocated = set(), 0, 0, [], {}
    cache = archive_dir or root / "outputs/release_downloads"
    tar_handles = {}
    try:
        for relative, entry in manifest["files"].items():
            if not all_records and not (relative.startswith("artifacts/orca_benchmark_suite/S12/") or
                    relative in ("artifacts/orca_benchmark_suite/S10/inputs/model.xml",
                                 "artifacts/orca_benchmark_suite/S09/inputs/reference.npz")):
                continue
            if "storage" in entry:
                source = contained(release, entry["storage"])
                if entry["sha256"] not in verified:
                    if not source.is_file() or digest(source) != entry["sha256"]:
                        raise ValueError("Corrupt release input: " + relative)
                    verified.add(entry["sha256"])
            elif not trajectories:
                continue
            else:
                name = entry["archive"]
                archive = contained(cache, name)
                info = manifest["archives"][name]
                if name not in tar_handles:
                    cache.mkdir(parents=True, exist_ok=True)
                    if not archive.exists():
                        partial = archive.with_suffix(".partial")
                        print("Download", info["url"], flush=True)
                        urllib.request.urlretrieve(info["url"], partial)
                        if digest(partial) != info["sha256"]:
                            raise ValueError("Downloaded archive hash mismatch")
                        partial.replace(archive)
                    if digest(archive) != info["sha256"]:
                        raise ValueError("Archive hash mismatch: " + name)
                    tar_handles[name] = tarfile.open(archive)
                handle = tar_handles[name]
                member = handle.getmember(entry["member"])
                if not member.isfile() or member.size != entry["bytes"]:
                    raise ValueError("Invalid trace archive member")
                source = None
            if verify_only and source is not None:
                continue
            target = contained(root, relative)
            content = None
            if source is None:
                content = handle.extractfile(member).read()
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError("Trace hash mismatch: " + relative)
                if verify_only:
                    continue
            elif target.suffix == ".xml" and target.name in ("model.xml", "cube.xml"):
                content = relocated_xml(source, manifest, release)
                relocated[relative] = dict(original_sha256=entry["sha256"],
                    runtime_sha256=hashlib.sha256(content).hexdigest(), change="asset paths only")
            if target.exists() or target.is_symlink():
                expected = hashlib.sha256(content).hexdigest() if content is not None else entry["sha256"]
                if target.is_file() and digest(target) in (entry["sha256"], expected):
                    existing += 1
                    continue
                conflicts.append(relative)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if content is not None:
                # A private temporary file prevents partial restoration on interruption.
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
                    f.write(content)
                    temporary = Path(f.name)
                temporary.replace(target)
            else:
                target.symlink_to(os.path.relpath(source.resolve(), target.parent))
            restored += 1
    finally:
        for handle in tar_handles.values():
            handle.close()
    report = dict(restored=restored, existing=existing, conflicts=conflicts,
                  verified_unique_payloads=len(verified), relocated=relocated)
    if not verify_only:
        out = root / "outputs/release_restore.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2)+"\n")
    if conflicts:
        raise RuntimeError("Refused to overwrite local files: " + str(conflicts[:10]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Restore historical records as well as S12")
    parser.add_argument("--trajectories", action="store_true", help="Download and restore full trace archives")
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    report = restore(all_records=args.all, trajectories=args.trajectories,
                     archive_dir=args.archive_dir, verify_only=args.verify_only)
    print(json.dumps({**report, "relocated": len(report["relocated"]),
                      "detail": "outputs/release_restore.json" if not args.verify_only else None}, indent=2))


if __name__ == "__main__":
    main()
