"""Conservatively prune only untracked intermediate legacy cube PPO checkpoints."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOTS = (ROOT / "Aero-Hand/logs", ROOT / "Aero-Hand/sim_rl/mujoco_playground/logs")
OUT = ROOT / "artifacts/maintenance/legacy_ppo_cleanup_20261009"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_files():
    result = set()
    for repo in (ROOT, ROOT / "Aero-Hand", ROOT / "Aero-Hand/sim_rl/mujoco_playground"):
        raw = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"])
        result.update((repo / p.decode()).absolute() for p in raw.split(b"\0") if p)
    return result


def references():
    steps, full, sources = set(), defaultdict(set), []
    extensions = {".py", ".md", ".json", ".jsonl", ".sh", ".yaml", ".yml", ".toml", ".txt"}
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "jax_cache", "node_modules", ".venv"}
                   and not d.isdigit() and not (Path(directory) / d).is_symlink()]
        base = Path(directory)
        if base.is_relative_to(OUT):
            dirs[:] = []
            continue
        for name in files:
            p = base / name
            if p.suffix not in extensions or p.is_symlink():
                continue
            text = p.read_text(errors="replace")
            if "checkpoint" not in text.lower() and not re.search(r"\b0\d{11}\b", text):
                continue
            local = set(re.findall(r"\b0\d{11}\b", text))
            steps.update(local)
            for run, step in re.findall(r"(Aero[^/\s\"'`<>]+)/checkpoints/(\d+)", text):
                full[run].add(int(step))
            if local:
                sources.append(str(p.relative_to(ROOT)))
    return {int(s) for s in steps}, full, sources


def metrics(run):
    values = defaultdict(list)
    path = run / "metrics.csv"
    source = "metrics.csv"
    if path.exists():
        with path.open() as stream:
            for row in csv.DictReader(stream):
                try:
                    step = int(float(row.get("num_steps", row.get("step", ""))))
                except (ValueError, TypeError):
                    continue
                for key, raw in row.items():
                    if key is None or not key.startswith("eval/") or "std" in key:
                        continue
                    if not ("episode_reward" in key or "episode_diagnostic" in key or key == "eval/avg_episode_length"):
                        continue
                    try:
                        value = float(raw)
                        if math.isfinite(value):
                            values[key].append((step, value))
                    except (ValueError, TypeError):
                        pass
    if not any(values.values()):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = list(run.glob("events.out.tfevents.*"))
        if events:
            source = "TensorBoard scalar events"
            accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags()["scalars"]:
                if tag.startswith("eval/") and "std" not in tag and (
                    "episode_reward" in tag or "episode_diagnostic" in tag or tag == "eval/avg_episode_length"
                ):
                    values[tag] = [(v.step, v.value) for v in accumulator.Scalars(tag) if math.isfinite(v.value)]
    return {k: sorted(v) for k, v in values.items() if v}, source


def preserved_steps(checkpoints, metrics_by_tag, mentioned):
    steps = sorted(checkpoints)
    kept = defaultdict(list)
    for step, label in ((steps[0], "first"), (steps[-1], "last")):
        kept[step].append(label)
    for fraction in (.25, .5, .75):
        kept[steps[round((len(steps)-1)*fraction)]].append("trajectory_landmark")
    for step in set(steps) & set(mentioned):
        kept[step].append("referenced_step")
    for tag, data in metrics_by_tag.items():
        # Preserve both extremes of every evaluation component, not only total reward.
        # This retains important failure modes without treating reward as success.
        for row in (min(data, key=lambda r:r[1]), max(data, key=lambda r:r[1])):
            step = row[0]
            before = [s for s in steps if s <= step]
            after = [s for s in steps if s >= step]
            for near in set(([before[-1]] if before else []) + ([after[0]] if after else [])):
                kept[near].append("metric_extreme:" + tag)
    return dict(kept)


def describe_files(directory):
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink inside checkpoint: " + str(path))
        if path.is_file():
            st = path.stat()
            result.append(dict(path=str(path.relative_to(ROOT)), bytes=st.st_size,
                               mtime_ns=st.st_mtime_ns, blocks=st.st_blocks,
                               sha256=sha(path)))
    return result


def make_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    tracked = tracked_files()
    mentioned, exact, reference_sources = references()
    runs, candidates, kept_count = [], [], 0
    for base in LOG_ROOTS:
        for run in sorted(base.iterdir()):
            if (not run.is_dir() or run.is_symlink() or not run.name.startswith("AeroCubeGrasp")
                    or not re.search(r"-20260[34]\d{2}-", run.name)
                    or any(s in run.name.lower() for s in ("residual", "prior", "dexmv", "guided", "demo"))):
                continue
            checkpoint_root = run / "checkpoints"
            checkpoints = {int(p.name): p for p in checkpoint_root.glob("*") if p.is_dir() and p.name.isdigit()}
            if not checkpoints:
                continue
            by_tag, source = metrics(run)
            info = dict(run=str(run.relative_to(ROOT)), checkpoints=len(checkpoints), metrics_source=source,
                        metrics={k: dict(first=v[0], last=v[-1], best=max(v,key=lambda x:x[1]),
                                         worst=min(v,key=lambda x:x[1]), evaluations=len(v)) for k,v in by_tag.items()})
            protected = preserved_steps(checkpoints, by_tag, mentioned | exact.get(run.name, set()))
            if len(checkpoints) <= 5 or not by_tag:
                protected = {s:["small_run_or_missing_metrics"] for s in checkpoints}
            for step, path in checkpoints.items():
                files = [p for p in path.rglob("*") if p.is_file()]
                if path.is_symlink() or any(p.is_symlink() for p in path.rglob("*")):
                    protected.setdefault(step, []).append("symlink")
                if any(p.absolute() in tracked for p in files):
                    protected.setdefault(step, []).append("git_tracked")
                if any(time.time() - p.stat().st_mtime < 86400 for p in files):
                    protected.setdefault(step, []).append("recently_modified")
            removed = []
            for step, path in sorted(checkpoints.items()):
                if step in protected:
                    continue
                files = describe_files(path)
                entry = dict(directory=str(path.relative_to(ROOT)), step=step, files=files,
                             bytes=sum(f["bytes"] for f in files), blocks=sum(f["blocks"] for f in files),
                             reason="Untracked legacy intermediate; first/last, metric extrema, landmarks and references retained")
                candidates.append(entry)
                removed.append(step)
            info.update(kept={str(k):v for k,v in protected.items()}, proposed_delete=removed)
            kept_count += len(protected)
            runs.append(info)
    evidence = {}
    for base in LOG_ROOTS:
        for path in base.rglob("*"):
            if path.is_file() and "checkpoints" not in path.relative_to(base).parts and (
                path.name == "metrics.csv" or path.name.startswith("events.out.tfevents.")):
                evidence[str(path.relative_to(ROOT))] = sha(path)
    plan = dict(version=1, date="2026-10-09", roots=[str(p.relative_to(ROOT)) for p in LOG_ROOTS],
                candidate_directories=candidates, runs=runs, protected_checkpoint_count=kept_count,
                scanned_runs=len(runs), reference_sources=reference_sources,
                preserved_log_sha256=evidence,
                planned_bytes=sum(c["bytes"] for c in candidates),
                planned_allocated_bytes=sum(c["blocks"]*512 for c in candidates),
                policy="No entire runs, tracked files, formal experiments, archives, datasets, meshes, or logs deleted")
    target = OUT / "plan.json"
    if target.exists():
        raise FileExistsError("Review existing plan instead of overwriting")
    target.write_text(json.dumps(plan, indent=2)+"\n")
    print(json.dumps({k:plan[k] for k in ("scanned_runs", "protected_checkpoint_count", "planned_bytes", "planned_allocated_bytes")}))
    print("candidate_directories", len(candidates))


def apply_plan():
    if (OUT / "execution.jsonl").exists():
        raise FileExistsError("Execution record already exists; do not repeat deletion")
    plan = json.loads((OUT / "plan.json").read_text())
    tracked = tracked_files()
    # Revalidate every file in the entire plan before touching the first checkpoint.
    for entry in plan["candidate_directories"]:
        directory = ROOT / entry["directory"]
        if (directory.is_symlink() or not any(directory.is_relative_to(r) for r in LOG_ROOTS)
                or directory.parent.name != "checkpoints" or not directory.name.isdigit()):
            raise ValueError("Outside deletion scope")
        current = describe_files(directory)
        if current != entry["files"] or any((ROOT / f["path"]).absolute() in tracked for f in current):
            raise ValueError("Checkpoint changed since review: " + str(directory))
    before = shutil.disk_usage(ROOT).free
    deleted_bytes = 0
    with (OUT / "execution.jsonl").open("x") as stream:
        for entry in plan["candidate_directories"]:
            directory = ROOT / entry["directory"]
            # The exact reviewed directory is the only deletion target.
            shutil.rmtree(directory)
            deleted_bytes += entry["bytes"]
            stream.write(json.dumps(dict(directory=entry["directory"], deleted=True, bytes=entry["bytes"]))+"\n")
            stream.flush()
    for rel, expected in plan["preserved_log_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError("Preserved log changed: " + rel)
    kept = 0
    for run in plan["runs"]:
        for step in run["kept"]:
            if not (ROOT / run["run"] / "checkpoints" / f"{int(step):012d}").is_dir():
                raise ValueError("Expected retained checkpoint missing")
            kept += 1
    result = dict(deleted_checkpoints=len(plan["candidate_directories"]), deleted_bytes=deleted_bytes,
                  retained_checkpoints=kept, log_files_unchanged=len(plan["preserved_log_sha256"]),
                  free_before=before, free_after=shutil.disk_usage(ROOT).free,
                  note="Actual free-space change can include other running programs; logical removed size is exact")
    (OUT / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-reviewed-plan", action="store_true")
    args = parser.parse_args()
    apply_plan() if args.apply_reviewed_plan else make_plan()
