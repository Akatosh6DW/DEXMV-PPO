"""Replace byte-identical data files with hard links; paths and contents are unchanged.

Only immutable binary data under artifacts/ and deliverables/ is considered.
Every target is re-verified byte-by-byte against its keeper immediately before
an atomic replace, and each action is appended to a JSONL log.
"""

import argparse
import filecmp
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".npz", ".mjb", ".stl", ".obj"}
ALLOWED = ("artifacts/", "deliverables/")
EXCLUDED = ("artifacts/orca_benchmark_suite/S09/",)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def eligible(rel, now, min_age):
    if not rel.startswith(ALLOWED) or rel.startswith(EXCLUDED):
        return False
    if Path(rel).suffix.lower() not in EXTENSIONS:
        return False
    p = ROOT / rel
    return p.is_file() and not p.is_symlink() and now - p.stat().st_mtime > min_age


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", type=Path, help="JSON list of {size, sha256, paths}")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--min-age-hours", type=float, default=24)
    a = ap.parse_args()
    now, min_age = time.time(), a.min_age_hours * 3600
    planned = linked = skipped = 0
    with open(a.log, "a") as log:
        for g in json.loads(a.groups.read_text()):
            rels = [str(Path(p).resolve().relative_to(ROOT)) for p in g["paths"]]
            rels = [r for r in rels if eligible(r, now, min_age)]
            if len(rels) < 2:
                continue
            # Keeper: oldest file, preferring the original under artifacts/.
            rels.sort(key=lambda r: (os.stat(ROOT / r).st_mtime, not r.startswith("artifacts/"), r))
            keeper = ROOT / rels[0]
            if keeper.stat().st_size != g["size"]:
                skipped += len(rels) - 1
                continue
            if a.execute and digest(keeper) != g["sha256"]:
                skipped += len(rels) - 1
                continue
            for rel in rels[1:]:
                target = ROOT / rel
                ks, ts = keeper.stat(), target.stat()
                if (ks.st_dev, ks.st_ino) == (ts.st_dev, ts.st_ino):
                    continue
                planned += g["size"]
                if not a.execute:
                    continue
                if ts.st_size != ks.st_size or not filecmp.cmp(keeper, target, shallow=False):
                    skipped += 1
                    continue
                tmp = target.with_name(target.name + ".dedupe_tmp")
                os.link(keeper, tmp)
                os.replace(tmp, target)
                linked += g["size"]
                log.write(json.dumps(dict(target=rel, keeper=rels[0], size=g["size"], sha256=g["sha256"],
                                          target_mtime_before=ts.st_mtime, target_mode_before=oct(ts.st_mode),
                                          time=time.strftime("%F %T")), ensure_ascii=False) + "\n")
    print(f"planned {planned/1e9:.2f} GB, linked {linked/1e9:.2f} GB, skipped {skipped} files")


if __name__ == "__main__":
    main()
