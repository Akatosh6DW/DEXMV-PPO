#!/usr/bin/env python3
"""Convert HIC sequence 20 motion fits into metric hand/object trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_motion(path):
    blocks = {}
    name = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            values = [float(v) for v in line.split()]
        except ValueError:
            name = line.strip()
            if name in blocks:
                raise ValueError("Duplicate bone: " + name)
            blocks[name] = []
            continue
        if name is None or len(values) != 12:
            raise ValueError("Invalid motion row in " + str(path))
        blocks[name].append(values)
    result = {}
    for name, rows in blocks.items():
        a = np.asarray(rows, dtype=np.float64)
        t = np.broadcast_to(np.eye(4), (len(a), 4, 4)).copy()
        t[:, :3, :3] = a[:, :9].reshape(-1, 3, 3)
        t[:, :3, 3] = a[:, 9:] / 1000.0
        r = t[:, :3, :3]
        if (not np.isfinite(t).all()
                or not np.allclose(np.linalg.det(r), 1, atol=1e-4)
                or not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-4)):
            raise ValueError("Invalid rotation for " + name)
        result[name] = t
    if len({len(v) for v in result.values()}) != 1:
        raise ValueError("Mismatched bone frame counts")
    return result


def read_skeleton(path):
    lines = path.read_text().splitlines()
    if (len(lines) - 1) % 3:
        raise ValueError("Invalid skeleton")
    return int(lines[0]), {
        lines[i + 1]: (lines[i], float(lines[i + 2]) / 1000.0)
        for i in range(1, len(lines), 3)
    }


def convert(source, fps):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    motion = read_motion(source / "Model_Hand_R.MOTION")
    n, skeleton = read_skeleton(source / "Model_Hand_R.SKEL")
    if n != len(motion["Lower Arm R"]):
        raise ValueError("Skeleton/motion frame count mismatch")
    joints = np.empty((n, 21, 3), dtype=np.float64)

    def origin(name):
        return motion[name][:, :3, 3]

    def endpoint(name):
        return origin(name) + motion[name][:, :3, 0] * skeleton[name][1]

    joints[:, 0] = endpoint("Lower Arm R")
    # HIC counts little/ring/middle/index/thumb; output counts thumb first.
    joints[:, 1:5] = np.stack([
        origin("RFing.4_Hand"), origin("RFing.4_0"),
        origin("RFing.4_1"), endpoint("RFing.4_1"),
    ], axis=1)
    for out_finger, hic_finger in enumerate([3, 2, 1, 0], start=1):
        prefix = "RFing.%d_" % hic_finger
        joints[:, 1 + 4 * out_finger:5 + 4 * out_finger] = np.stack([
            origin(prefix + "0"), origin(prefix + "1"),
            origin(prefix + "2"), endpoint(prefix + "2"),
        ], axis=1)
    # Validate axis and matrix conventions against observed child origins.
    # Metacarpal roots attach off the arm axis; verify only intra-finger links.
    chain_errors = []
    for f in range(4):
        for j in [0, 1]:
            a, b = "RFing.%d_%d" % (f, j), "RFing.%d_%d" % (f, j + 1)
            chain_errors.append(np.max(np.linalg.norm(endpoint(a) - origin(b), axis=1)))
    if max(chain_errors) > 1e-4:
        raise ValueError("Motion axes do not match skeleton lengths")
    obj = next(iter(read_motion(source / "Model_Object_Cube.MOTION").values()))
    if len(obj) != n:
        raise ValueError("Hand/object frame mismatch")
    # OFF vertices are in the rigging frame, not centered at the bone origin.
    # Rigid skinning is M(t) @ inverse(M(0)); the cube OFF is centered at zero.
    object_bone_poses = obj.copy()
    obj = obj @ np.linalg.inv(obj[0])
    # Frame zero is the rigging pose, not a recorded video frame (HIC README).
    return {
        "hand_joints": joints[1:].astype(np.float32),
        "object_poses": obj[1:].astype(np.float32),
        "object_bone_poses": object_bone_poses[1:].astype(np.float32),
        "time": np.arange(n - 1, dtype=np.float64) / fps,
        "dt": np.array(1.0 / fps),
        "source_motion_frame": np.arange(1, n, dtype=np.int32),
        "source_video_frame": np.arange(n - 1, dtype=np.int32),
        "bone_consistency_max_m": np.array(max(chain_errors)),
    }


def preview(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from itertools import product
    fig = plt.figure(figsize=(13, 4))
    corners = np.array(list(product([-0.0265, 0.0265], repeat=3)))
    for k, idx in enumerate([0, len(data["time"]) // 2, len(data["time"]) - 1]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        j = data["hand_joints"][idx]
        for f in range(5):
            chain = j[[0] + list(range(1 + 4*f, 5 + 4*f))]
            ax.plot(*chain.T, "o-", markersize=3)
        t = data["object_poses"][idx]
        c = corners @ t[:3, :3].T + t[:3, 3]
        for i in range(8):
            for l in range(i + 1, 8):
                if np.sum(corners[i] != corners[l]) == 1:
                    ax.plot(*c[[i, l]].T, color="black", alpha=0.5)
        center = np.mean(np.concatenate([j, c]), axis=0)
        for setter, value in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
            setter(value - .14, value + .14)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title("HIC20 video frame %d" % idx)
        ax.set_xlabel("camera x (m)"); ax.set_ylabel("camera y (m)"); ax.set_zlabel("camera z (m)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("artifacts/cube20/hic20/models_motion/20"))
    p.add_argument("--output", type=Path, default=Path("artifacts/cube20/hic20/prepared"))
    p.add_argument("--fps", type=float, required=True, help="Verify against the source video before setting.")
    a = p.parse_args()
    if a.fps <= 0:
        raise ValueError("fps must be positive")
    data = convert(a.source, a.fps)
    a.output.mkdir(parents=True, exist_ok=True)
    hand_dir, object_dir = a.output / "hand_pose", a.output / "object_pose"
    hand_dir.mkdir(exist_ok=True); object_dir.mkdir(exist_ok=True)
    for i, (j, t) in enumerate(zip(data["hand_joints"], data["object_poses"])):
        np.save(hand_dir / ("joints_%d.npy" % (i + 1)), j)
        np.save(object_dir / ("%d.npy" % (i + 1)), t[:3])
    meta = {
        "source_page": "https://hic.is.tue.mpg.de/download.html",
        "sequence": 20, "description": "Moving a Cube with one hand",
        "source_type": "published video-tracker motion fits; not raw-video inference rerun",
        "source_frames": len(data["time"]), "fps": a.fps,
        "units": "meters", "coordinates": "original camera coordinates",
        "object_pose_convention": "OFF mesh transform M(t) @ inverse(M(0)); not raw bone origin",
        "joint_order": "wrist, thumb CMC/MCP/IP/tip, index/middle/ring/little MCP/PIP/DIP/tip",
        "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.source.glob("*.MOTION"))},
        "bone_consistency_max_m": float(data["bone_consistency_max_m"]),
    }
    (a.output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.savez_compressed(a.output / "human_cube.npz", **data, metadata_json=np.array(json.dumps(meta)))
    preview(data, a.output / "source_geometry.png")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
