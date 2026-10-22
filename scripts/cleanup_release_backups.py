"""Remove two redundant report ZIPs after checking or retaining every member."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ("Orca_S02_S04_Experiments.zip", "汇报资料_S05-S08_2026-10-12.zip")
AUDIT = ROOT / "artifacts/maintenance/release_cleanup_20261023"
UNIQUE = ROOT / "deliverables/历史归档独有资料_2026-10-23"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        AUDIT.mkdir(parents=True, exist_ok=False)
    report = dict(executed=args.execute, free_before=shutil.disk_usage(ROOT).free, archives=[])
    for name in ARCHIVES:
        path = ROOT / "deliverables" / name
        record = dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                      sha256=sha(path), members=[], removed=False)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe archive path: " + member.filename)
                payload = archive.read(member)
                digest = hashlib.sha256(payload).hexdigest()
                existing = ROOT / "deliverables" / member.filename
                symlink = stat.S_ISLNK(member.external_attr >> 16)
                same = (existing.is_symlink() and existing.readlink().as_posix().encode() == payload
                        if symlink else existing.is_file() and sha(existing) == digest)
                destination = existing
                if not same:
                    destination = UNIQUE / path.stem / member.filename
                    if symlink:
                        destination = destination.with_name(destination.name + ".symlink.txt")
                    if args.execute:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("xb") as stream:
                            stream.write(payload)
                        if sha(destination) != digest:
                            raise ValueError("Retained file verification failed")
                record["members"].append(dict(path=member.filename, bytes=member.file_size,
                    sha256=digest, symlink=symlink, identical_existing=same,
                    retained_as=str(destination.relative_to(ROOT))))
        report["archives"].append(record)
        if args.execute:
            # Persist the complete inventory before unlinking the redundant container.
            (AUDIT / "archives.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
            if sha(path) != record["sha256"]:
                raise ValueError("Archive changed during verification")
            path.unlink()
            record["removed"] = True
            (AUDIT / "archives.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
        print(json.dumps(dict(archive=name, members=len(record["members"]),
            preserved_unique=sum(not x["identical_existing"] for x in record["members"]),
            bytes=record["bytes"], removed=record["removed"]), ensure_ascii=False), flush=True)
    report["free_after"] = shutil.disk_usage(ROOT).free
    if args.execute:
        (AUDIT / "archives.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
