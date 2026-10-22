#!/usr/bin/env python3
"""Retarget DexMV 21-joint hand-pose files into ORCA hand v2 controls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


ORCA_CTRL_NAMES = [
    "right_wrist",
    "right_p-abd",
    "right_p-mcp",
    "right_p-pip",
    "right_r-abd",
    "right_r-mcp",
    "right_r-pip",
    "right_m-abd",
    "right_m-mcp",
    "right_m-pip",
    "right_i-abd",
    "right_i-mcp",
    "right_i-pip",
    "right_t-cmc",
    "right_t-abd",
    "right_t-mcp",
    "right_t-pip",
]

ORCA_CTRL_LOW = np.array([
    -1.1344640,
    -0.5235988, -0.4363323, -0.2617994,
    -0.4712389, -0.4363323, -0.2617994,
    -0.4712389, -0.4363323, -0.2617994,
    -0.4363323, -0.4363323, -0.2617994,
    -0.7853982, -0.3141593, -0.4363323, -0.2617994,
], dtype=np.float32)

ORCA_CTRL_HIGH = np.array([
    0.6108652,
    0.5235988, 1.7453293, 1.8675023,
    0.4712389, 1.7453293, 1.8675023,
    0.4712389, 1.7453293, 1.8675023,
    0.5235988, 1.7453293, 1.8675023,
    0.5759587, 0.9599311, 1.7453293, 1.8675023,
], dtype=np.float32)

ORCA_DEFAULT_CTRL = np.clip(np.zeros(17, dtype=np.float32), ORCA_CTRL_LOW, ORCA_CTRL_HIGH)
ORCA_ACTION_SCALE = np.full(17, 0.18, dtype=np.float32)

FINGER_JOINTS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def _natural_key(path: Path) -> list[object]:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.name)]


def _safe_norm(vec: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(vec, axis=axis) + 1e-8


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    denom = _safe_norm(a) * _safe_norm(b)
    return np.arccos(np.clip(dot / denom, -1.0, 1.0))


def _finger_closure(joints: np.ndarray, ids: list[int]) -> np.ndarray:
    p = joints[:, ids, :]
    b0 = p[:, 1] - p[:, 0]
    b1 = p[:, 2] - p[:, 1]
    b2 = p[:, 3] - p[:, 2]
    mcp_bend = _angle_between(b0, b1)
    pip_bend = _angle_between(b1, b2)
    return np.clip((0.55 * mcp_bend + 0.45 * pip_bend) / 1.45, 0.0, 1.0)


def _finger_spread(joints: np.ndarray, ids: list[int], scale: float = 2.0) -> np.ndarray:
    wrist = joints[:, 0]
    index_mcp = joints[:, 5]
    pinky_mcp = joints[:, 17]
    palm_x = index_mcp - pinky_mcp
    base_vec = joints[:, ids[0]] - wrist
    spread = np.sum(base_vec * palm_x, axis=1) / (_safe_norm(base_vec) * _safe_norm(palm_x))
    centered = spread - spread[0]
    return np.clip(centered * scale, -1.0, 1.0)


def _thumb_opposition(joints: np.ndarray) -> np.ndarray:
    wrist = joints[:, 0]
    index_mcp = joints[:, 5]
    pinky_mcp = joints[:, 17]
    thumb_tip = joints[:, 4]
    palm_x = index_mcp - pinky_mcp
    palm_y = 0.5 * (index_mcp + pinky_mcp) - wrist
    palm_normal = np.cross(palm_x, palm_y)
    thumb_vec = thumb_tip - wrist
    across = np.sum(thumb_vec * palm_x, axis=1) / (_safe_norm(thumb_vec) * _safe_norm(palm_x))
    into = np.sum(thumb_vec * palm_normal, axis=1) / (_safe_norm(thumb_vec) * _safe_norm(palm_normal))
    return np.clip(0.60 * ((across + 1.0) * 0.5) + 0.40 * ((into + 1.0) * 0.5), 0.0, 1.0)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        raise ValueError("--smoothing-window must be odd.")
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.empty_like(values)
    for i in range(values.shape[1]):
        out[:, i] = np.convolve(padded[:, i], kernel, mode="valid")
    return out


def _load_hand_joints(hand_dir: Path) -> tuple[np.ndarray, list[str]]:
    files = sorted(hand_dir.glob("joints_*.npy"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No joints_*.npy files found in {hand_dir}")
    joints = [np.load(path).astype(np.float32) for path in files]
    joint_seq = np.stack(joints, axis=0)
    if joint_seq.ndim != 3 or joint_seq.shape[1:] != (21, 3):
        raise ValueError(f"Expected DexMV hand joints shape (T, 21, 3), got {joint_seq.shape}")
    return joint_seq, [path.name for path in files]


def _load_object_pose_dir(path: Path | None, target_len: int) -> np.ndarray:
    if path is None:
        return np.zeros((0, 3, 4), dtype=np.float32)
    poses = []
    for item in sorted(path.glob("*.npy"), key=_natural_key):
        pose = np.load(item, allow_pickle=True)
        if pose.shape == () and pose.dtype == object:
            pose_dict = pose.item()
            if not pose_dict:
                continue
            pose = pose_dict[sorted(pose_dict.keys())[0]]
        poses.append(np.asarray(pose, dtype=np.float32))
    if not poses:
        return np.zeros((0, 3, 4), dtype=np.float32)
    object_poses = np.stack(poses, axis=0)
    if object_poses.shape[0] > target_len:
        return object_poses[:target_len]
    if object_poses.shape[0] < target_len:
        pad = np.repeat(object_poses[-1:, ...], target_len - object_poses.shape[0], axis=0)
        return np.concatenate([object_poses, pad], axis=0)
    return object_poses


def _amount_to_range(amount: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + np.clip(amount, 0.0, 1.0) * (high - low)


def _spread_to_range(spread: np.ndarray, low: float, high: float) -> np.ndarray:
    centered = 0.5 + 0.5 * np.clip(spread, -1.0, 1.0)
    return low + centered * (high - low)


def _hand_joints_to_orca_ctrl(joints: np.ndarray) -> np.ndarray:
    ctrl = np.repeat(ORCA_DEFAULT_CTRL[None, :], joints.shape[0], axis=0)

    curls = {name: _finger_closure(joints, ids) for name, ids in FINGER_JOINTS.items()}
    spreads = {name: _finger_spread(joints, ids) for name, ids in FINGER_JOINTS.items() if name != "thumb"}
    thumb_opp = _thumb_opposition(joints)

    # ORCA actuator order: wrist, pinky, ring, middle, index, thumb.
    for name, base_idx in [("pinky", 1), ("ring", 4), ("middle", 7), ("index", 10)]:
        ctrl[:, base_idx] = _spread_to_range(spreads[name], ORCA_CTRL_LOW[base_idx], ORCA_CTRL_HIGH[base_idx])
        ctrl[:, base_idx + 1] = _amount_to_range(curls[name], ORCA_CTRL_LOW[base_idx + 1], ORCA_CTRL_HIGH[base_idx + 1])
        ctrl[:, base_idx + 2] = _amount_to_range(curls[name] * 1.05, ORCA_CTRL_LOW[base_idx + 2], ORCA_CTRL_HIGH[base_idx + 2])

    thumb = curls["thumb"]
    ctrl[:, 13] = _amount_to_range(thumb_opp, ORCA_CTRL_LOW[13], ORCA_CTRL_HIGH[13])
    ctrl[:, 14] = _amount_to_range(thumb_opp, ORCA_CTRL_LOW[14], ORCA_CTRL_HIGH[14])
    ctrl[:, 15] = _amount_to_range(thumb, ORCA_CTRL_LOW[15], ORCA_CTRL_HIGH[15])
    ctrl[:, 16] = _amount_to_range(thumb * 1.05, ORCA_CTRL_LOW[16], ORCA_CTRL_HIGH[16])
    return np.clip(ctrl, ORCA_CTRL_LOW[None, :], ORCA_CTRL_HIGH[None, :]).astype(np.float32)


def build_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    hand_joints, joint_files = _load_hand_joints(args.hand_dir)
    ctrl = _hand_joints_to_orca_ctrl(hand_joints)
    ctrl = _moving_average(ctrl, args.smoothing_window)
    action = np.clip((ctrl - ORCA_DEFAULT_CTRL[None, :]) / ORCA_ACTION_SCALE[None, :], -1.0, 1.0)
    time = np.arange(ctrl.shape[0], dtype=np.float32) * np.float32(args.dt)
    object_poses = _load_object_pose_dir(args.object_pose_dir, ctrl.shape[0])
    metadata = {
        "source": str(args.hand_dir),
        "target_hand": "ORCA hand v2 right",
        "dt": float(args.dt),
        "retarget_type": "direct_dexmv_video_to_orca_ctrl",
        "mapping": {
            "wrist": "kept at neutral 0.0 until a palm-pose retargeter is added",
            "finger_abd": "DexMV finger base spread around the first-frame palm spread",
            "finger_mcp/pip": "DexMV per-finger bend angles mapped to ORCA joint ranges",
            "thumb_cmc/abd": "DexMV thumb opposition relative to palm plane",
            "thumb_mcp/pip": "DexMV thumb bend angles",
        },
        "joint_files_first_last": [joint_files[0], joint_files[-1]],
    }
    return {
        "dexmv_hand_joints": hand_joints.astype(np.float32),
        "orca_prior_ctrl": ctrl.astype(np.float32),
        "orca_prior_action": action.astype(np.float32),
        "orca_default_ctrl": ORCA_DEFAULT_CTRL.copy(),
        "orca_action_scale": ORCA_ACTION_SCALE.copy(),
        "orca_ctrl_low": ORCA_CTRL_LOW.copy(),
        "orca_ctrl_high": ORCA_CTRL_HIGH.copy(),
        "orca_ctrl_names": np.array(ORCA_CTRL_NAMES),
        "time": time,
        "dt": np.array(args.dt, dtype=np.float32),
        "object_poses": object_poses,
        "metadata_json": np.array(json.dumps(metadata, indent=2), dtype=np.str_),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hand-dir",
        type=Path,
        default=Path("DexMV/examples/retargeting_source/relocate_mustard_example_seq/hand_pose"),
        help="DexMV hand_pose directory containing joints_*.npy files.",
    )
    parser.add_argument("--object-pose-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ppo_bridge/demo_dataset_orca_direct.npz"),
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--smoothing-window", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **dataset)
    ctrl = dataset["orca_prior_ctrl"]
    print(f"Wrote {args.output}")
    print(f"frames={ctrl.shape[0]}, orca_prior_ctrl_shape={ctrl.shape}")
    print(f"orca_prior_ctrl_min={np.round(ctrl.min(axis=0), 4).tolist()}")
    print(f"orca_prior_ctrl_max={np.round(ctrl.max(axis=0), 4).tolist()}")


if __name__ == "__main__":
    main()

