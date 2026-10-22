"""Prune logged early Aero PPO intermediates; preserve endpoints and metric peaks."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "Aero-Hand/sim_rl/mujoco_playground/logs"
AUDIT = ROOT / "artifacts/maintenance/release_cleanup_20261023/legacy_checkpoints.json"
REFERENCE = re.compile(r"(Aero[^\s/\\\"']+/checkpoints/\d{6,12})(?!\d)")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def references():
    paths = set()
    for root in (ROOT / "docs", ROOT / "scripts", ROOT / "artifacts"):
        for suffix in ("*.json", "*.md", "*.py", "*.sh"):
            paths.update(root.rglob(suffix))
    for suffix in ("*.json", "*.md", "*.py", "*.sh"):
        paths.update((ROOT / "Aero-Hand").rglob(suffix))
    protected = set()
    for path in paths:
        if any(p in path.parts for p in (".git", "external_deps", "__pycache__", "maintenance")):
            continue
        if path.stat().st_size > 30_000_000:
            continue
        try:
            protected.update(REFERENCE.findall(path.read_text()))
        except (UnicodeError, OSError):
            continue
    return protected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and AUDIT.exists():
        raise FileExistsError(AUDIT)
    protected = references()
    report = dict(executed=args.execute, runs=[], removed=[],
                  policy="Only pre-May 2026 logged Aero runs; keep first, last, peak success/hold/reward and explicit references",
                  free_before=shutil.disk_usage(ROOT).free)
    candidates = []
    retained = {}
    for run in sorted(LOGS.iterdir()):
        if not re.match(r"Aero(?:Cube|Can).*-20260[34]\d{2}-", run.name):
            continue
        steps = sorted((p for p in (run / "checkpoints").glob("*")
                        if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
        metrics = run / "metrics.csv"
        if len(steps) < 3 or not metrics.exists():
            continue
        with metrics.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < 3 or "num_steps" not in rows[0]:
            continue
        observed = {}
        for row in rows:
            try:
                observed[int(float(row["num_steps"]))] = row
            except (ValueError, TypeError):
                pass
        if int(steps[-1].name) not in observed:
            continue
        keep = {steps[0], steps[-1]}
        columns = [c for c in rows[0] if not c.endswith("_std") and
                   ("success" in c or "hold" in c or c == "eval/episode_reward")]
        peaks = {}
        for column in columns:
            scored = []
            for step in steps:
                try:
                    value = float(observed[int(step.name)][column])
                    if math.isfinite(value):
                        scored.append((value, step))
                except (KeyError, ValueError, TypeError):
                    pass
            if scored:
                maximum = max(value for value, _ in scored)
                # Tied positive peaks are all retained, not arbitrarily ranked by reward.
                winners = [p for v, p in scored if v == maximum] if maximum != 0 else []
                keep.update(winners)
                peaks[column] = dict(value=maximum, steps=[p.name for p in winners])
        keep.update(p for p in steps if f"{run.name}/checkpoints/{p.name}" in protected)
        remove = [p for p in steps if p not in keep]
        if not remove:
            continue
        report["runs"].append(dict(run=run.name, metrics_sha256=sha(metrics), metric_peaks=peaks,
                                  kept=[p.name for p in sorted(keep)], removed=[p.name for p in remove]))
        candidates.extend(remove)
        for folder in keep:
            for path in folder.rglob("*"):
                if path.is_file():
                    retained[str(path.relative_to(ROOT))] = sha(path)
    for folder in candidates:
        if folder.is_symlink():
            raise ValueError("Refuse symlink checkpoint")
        files = []
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError("Refuse checkpoint with symlink")
            if path.is_file():
                files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path)))
        report["removed"].append(dict(folder=str(folder.relative_to(ROOT)), files=files))
    report.update(checkpoints=len(candidates), bytes=sum(f["bytes"] for r in report["removed"] for f in r["files"]),
                  retained_hashes=retained, protected_references=sorted(protected))
    if args.execute:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_text(json.dumps(report, indent=2)+"\n")
        for item in report["removed"]:
            for f in item["files"]:
                if sha(ROOT / f["path"]) != f["sha256"]:
                    raise ValueError("Checkpoint changed during cleanup")
            shutil.rmtree(ROOT / item["folder"])
        for path, expected in retained.items():
            if sha(ROOT / path) != expected:
                raise ValueError("Retained checkpoint changed")
        report.update(completed=True, free_after=shutil.disk_usage(ROOT).free)
        AUDIT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("runs", "removed", "retained_hashes", "protected_references")}), flush=True)
    print("affected_runs", len(report["runs"]), "preserved_files", len(retained), flush=True)


if __name__ == "__main__":
    main()
