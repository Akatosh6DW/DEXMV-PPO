"""Delete only authorized obsolete coarse-model rollout archives, with audit list."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ("orca_guidance_long/L01_train", "orca_guidance_long/T01_nominal",
       "orca_guidance_long/T02_offset", "orca_guidance_recheck", "orca_residual")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    candidates = sorted({f.resolve() for name in OLD for f in (ROOT / "artifacts" / name).rglob("trajectories.npz")
                         if f.stat().st_size > 1_000_000 and "/shared_cube_demos/" not in str(f)})
    # A frozen/new experiment may reference an old input. Keep exact references.
    protected = set()
    for directory in ("orca_formal", "orca_collision_refit", "orca_cube_acquisition"):
        for f in (ROOT / "artifacts" / directory).rglob("*.json"):
            if f.stat().st_size > 30_000_000:
                continue
            try:
                obj = json.loads(f.read_text())
            except (ValueError, UnicodeError):
                continue
            def strings(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, dict):
                    for k, v in value.items():
                        yield str(k)
                        yield from strings(v)
                elif isinstance(value, list):
                    for v in value:
                        yield from strings(v)
            for value in strings(obj):
                if value.endswith("trajectories.npz"):
                    path = Path(value)
                    protected.add((path if path.is_absolute() else ROOT / path).resolve())
    candidates = [f for f in candidates if f not in protected]
    report = dict(scope="Only obsolete coarse-model trajectories.npz > 1MB; preserve weights, metrics, sources, datasets, references and fine-model runs",
                  execute=args.execute, files=len(candidates), bytes=sum(f.stat().st_size for f in candidates),
                  protected=len(protected), free_before=shutil.disk_usage(ROOT).free)
    print(json.dumps(report), flush=True)
    if not args.execute:
        return
    output = ROOT / "artifacts/orca_cube_acquisition/cleanup_coarse_rollouts.json"
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for f in candidates:
        with f.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append(dict(path=str(f.relative_to(ROOT)), size=f.stat().st_size, sha256=digest))
    report["removed"] = rows
    output.write_text(json.dumps(report, indent=2) + "\n")
    for f in candidates:
        f.unlink()
    report.update(completed=True, free_after=shutil.disk_usage(ROOT).free)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "removed"}), flush=True)


if __name__ == "__main__":
    main()
