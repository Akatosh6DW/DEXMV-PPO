"""Check geometric penetration in saved active states, without resimulation."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from prepare_hic_orca_object import digest


def audit(model_path, trace_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    cube = model.geom("cube").id
    with np.load(trace_path) as packet:
        trace = dict(packet)
    penetration, pairs, peak = [], {}, None
    for episode, step in zip(*np.where(trace["active"])):
        for name in ("qpos", "qvel", "ctrl"):
            getattr(data, name)[:] = trace[name][episode, step]
        data.mocap_pos[:] = trace["mocap"][episode, step]
        mujoco.mj_forward(model, data)
        depth = 0.
        for contact in data.contact:
            if cube not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == cube else contact.geom1
            value = max(-float(contact.dist), 0.)
            name = model.geom(other).name
            pairs[name] = max(pairs.get(name, 0.), value)
            if peak is None or value > peak["depth_m"]:
                peak = dict(episode=int(episode), step=int(step), geom=name, depth_m=value)
            depth = max(depth, value)
        penetration.append(depth)
    if not penetration:
        raise ValueError("No active states")
    values = np.asarray(penetration)
    return dict(model_sha256=digest(model_path), trace_sha256=digest(trace_path),
                states=len(values), max_penetration_m=float(values.max()),
                p95_penetration_m=float(np.quantile(values, .95)),
                fraction_over_3mm=float(np.mean(values > .003)),
                fraction_over_5mm=float(np.mean(values > .005)), peak=peak,
                pair_max_penetration_m=pairs,
                scope="Native collision distances on saved float32 states; no resimulation or force equivalence claim; disabled self-collisions not checked")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.model, args.trace)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
