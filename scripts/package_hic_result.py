#!/usr/bin/env python3
"""Bundle verified HIC/PPO weights, prior, video and evaluation evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=root / "artifacts/cube20/hic20/eval_H02_16seeds")
    parser.add_argument("--output", type=Path, default=root / "artifacts/cube20/final_hic_bc_residual_ppo")
    args = parser.parse_args()
    evaluation = args.evaluation.resolve()
    report = json.loads((evaluation / "report.json").read_text())
    if not any(row["success_gt20s"] for row in report["episodes"]):
        raise ValueError("No verified >20 second episode")
    if not (evaluation / "best_rollout.mp4").is_file():
        raise FileNotFoundError("Missing verified rollout video")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve the existing bundle; choose an explicit new version")
    output.mkdir(parents=True)
    shutil.copytree(report["checkpoint"], output / "checkpoint")
    shutil.copy2(report["prior_path"], output / "demo_dataset.npz")
    for filename in ["best_rollout.mp4", "episodes.csv", "report.json", "trajectories.npz"]:
        shutil.copy2(evaluation / filename, output / filename)
    shutil.copy2(root / "artifacts/cube20/hic20/dexmv_bc/policy.pickle", output / "dexmv_bc_policy.pickle")
    config = json.loads((evaluation / "env_config.json").read_text())
    config["residual_prior"]["dataset_path"] = str(output / "demo_dataset.npz")
    (output / "env_config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(output / "env_config.json", output / "checkpoint/config.json")
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob("*")) if p.is_file()}
    if hashes["demo_dataset.npz"] != report["prior_sha256"]:
        raise ValueError("Prior does not match evaluated input")
    manifest = {"source_evaluation": str(evaluation), "source_checkpoint": report["checkpoint"],
                "blend_alpha": config["residual_prior"]["blend_alpha"],
                "target_hand": "Aero-Hand V2, NOT Orca", "success_count": sum(r["success_gt20s"] for r in report["episodes"]),
                "episode_count": len(report["episodes"]), "success_rate": report["success_rate"],
                "longest_continuous_hold_sec": max(r["continuous_unsupported_sec"] for r in report["episodes"]),
                "video_seed": max(report["episodes"], key=lambda r: r["continuous_unsupported_sec"])["seed"],
                "pipeline": "HIC fitted poses -> DexMV position retarget -> mjrl BC prior -> Aero mapping -> PPO continuation",
                "requirements": "Existing Aero-Hand code/assets and aero_rl environment; not a standalone hardware policy",
                "dexmv_bc_alone_physical_grasp_success": False, "sha256": hashes}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
