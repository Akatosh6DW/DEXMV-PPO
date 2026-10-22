"""Build full-action BC supervision from the same reference used by residual PPO."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from orca_residual_metrics import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    meta = json.loads((args.run/"run_config.json").read_text())
    if not meta["zero"] or meta["shape"] != "cube":
        raise ValueError("Use a cube prior-only rollout; do not silently add other demonstrations")
    source = args.run/"step_000000000/trajectories.npz"
    with np.load(source, allow_pickle=False) as z:
        trace = dict(z)
    model = mujoco.MjModel.from_xml_path(str(args.run/"model.xml"))
    low, high = model.actuator_ctrlrange.T
    _, _, keep = summarize(trace, "residual", meta["bound"], low, high)
    if not keep.any():
        raise ValueError("No successful complete cube demonstration; preserve and diagnose failures")
    obs = trace["obs"][keep].copy()
    # Pure/BC controllers have zeros in the reference feature, but share phase.
    obs[:, :, -18:-1] = 0
    action = np.clip(2*(trace["desired_ctrl"][keep]-low)/(high-low)-1, -1, 1)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/"trajectories.npz", obs=obs, action=action)
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  reference_sha256=meta["reference_sha256"], source_provenance=meta["reference_provenance"],
                  selected_episode_indices=np.flatnonzero(keep).tolist(),
                  excluded_episode_indices=np.flatnonzero(~keep).tolist(),
                  human_video_count=1, observation_dim=113, action_dim=17,
                  source_task="cube only; unchanged demonstrations transferred to sphere/cuboid",
                  selection="continuous clean 30s grasp and valid final frame at 40s",
                  steps=int(obs.shape[0]*obs.shape[1]))
    (args.output/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
