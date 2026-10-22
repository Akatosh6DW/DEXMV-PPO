"""Select complete successful demonstration episodes and retain provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.source) as z:
        raw = {k: z[k] for k in z.files}
    keep = raw["alive"].all(axis=1) & (raw["hold_sec"].max(axis=1) >= 30)
    if not np.any(keep):
        raise ValueError("No complete 30 s demonstrations; do not train on failed tails")
    selected = {k: v[keep] for k, v in raw.items()}
    assert np.isfinite(selected["obs"]).all()
    assert np.isfinite(selected["action"]).all()
    np.savez_compressed(args.output / "trajectories.npz", **selected)
    report = dict(source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  source_provenance=json.loads((args.source.parent / "report.json").read_text()),
                  selected_episode_indices=np.flatnonzero(keep).tolist(),
                  excluded_episode_indices=np.flatnonzero(~keep).tolist(),
                  raw_control_steps=int(raw["obs"].shape[0] * raw["obs"].shape[1]),
                  selected_control_steps=int(selected["obs"].shape[0] * selected["obs"].shape[1]),
                  selection="alive throughout 40s and continuous hold >=30s; all failures retained in source artifact",
                  human_video_count=1)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "source_provenance"}), flush=True)


if __name__ == "__main__":
    main()
