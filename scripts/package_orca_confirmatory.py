#!/usr/bin/env python3
"""Archive all paired policies and their evidence, not only successful seeds."""

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/orca30/confirmatory_f01"
DESTINATION = ROOT / "artifacts/orca30/final_f01"


def main():
    reports = json.loads((SOURCE / "heldout_summary.json").read_text())
    identities = {(r["arm"], r["seed"], r["kind"]) for r in reports}
    expected = {(arm, seed, kind) for arm in ("scratch", "bc")
                for seed in range(5) for kind in ("selected", "terminal")}
    if identities != expected or len(reports) != 20:
        raise ValueError("All ten selected and terminal evaluations are required")
    frozen = json.loads((SOURCE / "manifest.json").read_text())
    for name, digest in frozen["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {name}")
    DESTINATION.mkdir(parents=True, exist_ok=False)
    for arm in ("scratch", "bc"):
        for seed in range(5):
            src = SOURCE / f"{arm}_seed{seed}"
            dst = DESTINATION / src.name
            dst.mkdir()
            for filename in ("env_config.json", "run_config.json", "model.xml", "metrics.csv", "training_summary.json"):
                shutil.copy2(src / filename, dst / filename)
            for report in (r for r in reports if r["arm"] == arm and r["seed"] == seed):
                step = report["step"]
                checkpoint = dst / f"step_{step:09d}"
                checkpoint.mkdir(exist_ok=True)
                shutil.copy2(src / checkpoint.name / "params.pkl", checkpoint / "params.pkl")
                evaluation = src / f"heldout_step_{step:09d}" / "report.json"
                shutil.copy2(evaluation, dst / f"{report['kind']}_heldout.json")
    shutil.copytree(SOURCE / "analysis", DESTINATION / "analysis")
    shutil.copytree(SOURCE / "videos", DESTINATION / "videos")
    shutil.copytree(SOURCE / "mesh_alignment", DESTINATION / "mesh_alignment")
    shutil.copy2(SOURCE / "heldout_summary.json", DESTINATION / "heldout_summary.json")
    shutil.copy2(SOURCE / "software_versions.json", DESTINATION / "software_versions.json")
    shutil.copytree(ROOT / "artifacts/orca30/video3d_demos_v1", DESTINATION / "demonstrations")
    scripts = DESTINATION / "source_snapshot"
    scripts.mkdir()
    for name in frozen["source_sha256"]:
        shutil.copy2(ROOT / name, scripts / Path(name).name)
    manifest = dict(source=str(SOURCE), frozen_protocol=frozen,
                    requires="Existing aero_rl environment and this workspace's Orca model assets",
                    traces="All full evaluation traces remain in the source experiment directory",
                    files={str(p.relative_to(DESTINATION)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in DESTINATION.rglob("*") if p.is_file()})
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(DESTINATION)


if __name__ == "__main__":
    main()
