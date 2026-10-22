"""Measure cross-finger proxy intersections even when collision masks disable them."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from prepare_hic_orca_object import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--all-episodes", action="store_true")
    p.add_argument("--stride", type=int, default=5)
    args = p.parse_args()
    if args.stride < 1:
        p.error("Positive stride required")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    groups = {f: [i for i in range(model.ngeom) if model.geom(i).name.startswith(f"orca_{f}_")
                  and model.geom_contype[i] != 0] for f in fingers}
    with np.load(args.trace) as z:
        qpos = z["qpos"]
        active = z["active"].astype(bool) if "active" in z else np.ones(qpos.shape[:-1], bool)
        if qpos.ndim == 2:
            qpos, active = qpos[None], active[None]
        if not args.all_episodes:
            qpos, active = qpos[:1], active[:1]
    buffer = np.empty(6)
    worst = []
    pairs = {}
    frames = [(episode, int(frame)) for episode in range(len(qpos))
              for frame in np.unique(np.r_[np.arange(0, qpos.shape[1], args.stride), qpos.shape[1] - 1])
              if active[episode, frame]]
    if not frames:
        raise ValueError("No active sampled states")
    for episode, frame in frames:
        data.qpos[:] = qpos[episode, frame]
        mujoco.mj_forward(model, data)
        record = dict(episode=episode, frame=int(frame), depth_m=0., pair=None)
        for first in range(len(fingers)):
            for second in range(first + 1, len(fingers)):
                for a in groups[fingers[first]]:
                    for b in groups[fingers[second]]:
                        if np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b]) > model.geom_rbound[a] + model.geom_rbound[b]:
                            continue
                        distance = mujoco.mj_geomDistance(model, data, a, b, .005, buffer)
                        if distance < 0:
                            names = (model.geom(a).name, model.geom(b).name)
                            key = " | ".join(names)
                            if key not in pairs or -distance > pairs[key]["depth_m"]:
                                pairs[key] = dict(episode=episode, frame=int(frame), depth_m=float(-distance), pair=list(names),
                                                  includes_base_proxy=any("_base_" in name for name in names))
                        if -distance > record["depth_m"]:
                            record.update(depth_m=float(-distance), pair=[model.geom(a).name, model.geom(b).name])
        worst.append(record)
    result = dict(model_sha256=digest(args.model), trace_sha256=digest(args.trace),
                  episodes=len(qpos), stride=args.stride, active_only=True,
                  worst=max(worst, key=lambda r: r["depth_m"]), samples=worst,
                  pairs=sorted(pairs.values(), key=lambda r: -r["depth_m"]),
                  worst_nonbase=max((r for r in pairs.values() if not r["includes_base_proxy"]),
                                    key=lambda r: r["depth_m"], default=None),
                  note="Sampled cross-finger proxy intersections, not physical forces. Same-finger and finger-palm pairs are not tested.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("worst", "worst_nonbase")}), flush=True)


if __name__ == "__main__":
    main()
