"""Freeze a complete development reference bank; never filter by success."""

import argparse
import json
from pathlib import Path

import numpy as np

from retarget_s09_reference import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not (a.input/"manifest.json").is_file():
        raise ValueError("Reference development is incomplete")
    folders = sorted(a.input.glob("pose_*"))
    packets = []
    for folder in folders:
        with np.load(folder/"reference.npz") as z:
            packets.append(dict(z))
    if len(packets) < 3:
        raise ValueError("Need multiple orientations, not a single fixed reference")
    for packet in packets:
        np.testing.assert_array_equal(packet["time"], packets[0]["time"])
        np.testing.assert_allclose(packet["object_position"], [.08,.055,.25])
        np.testing.assert_array_equal(packet["initial_ctrl"], np.zeros(17))
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/"model.xml").write_bytes((a.input/"model.xml").read_bytes())
    np.savez_compressed(a.output/"reference.npz", time=packets[0]["time"],
        ctrl=np.stack([r["ctrl"] for r in packets]),
        object_quaternion=np.stack([r["object_quaternion"] for r in packets]),
        object_position=np.stack([r["object_position"] for r in packets]), initial_ctrl=np.zeros(17))
    meta = dict(input=str(a.input), pose_count=len(packets), success_filter=False,
        model_sha256=digest(a.output/"model.xml"), reference_sha256=digest(a.output/"reference.npz"),
        sources={str(f/"reference.npz"):digest(f/"reference.npz") for f in folders},
        base_selection="nearest undirected cylinder axis at reset; no test-success oracle",
        origin="DexMV sequence96 object-relative fingertips, constrained IK and bounded simulation correction")
    (a.output/"bank_manifest.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
