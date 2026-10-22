#!/usr/bin/env python3
"""Retarget DexMV hand-pose files directly into Aero-Hand V2 prior controls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


AERO_ACTION_NAMES = [
    "thumb_rot",
    "thumb_flex",
    "index",
    "middle",
    "ring",
    "pinky",
]

AERO_QPOS_NAMES = [
    "index_mcp",
    "index_pip",
    "middle_mcp",
    "middle_pip",
    "ring_mcp",
    "ring_pip",
    "pinky_mcp",
    "pinky_pip",
    "thumb_cmc_abd",
    "thumb_cmc_flex",
    "thumb_mcp",
]

AERO_CTRL_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
AERO_CTRL_HIGH = np.array([1.38, 0.79, 1.40, 1.40, 1.40, 1.40], dtype=np.float32)
AERO_DEFAULT_CTRL = np.array([1.38, 0.33, 0.55, 0.66, 1.40, 1.40], dtype=np.float32)
AERO_ACTION_SCALE = np.array([0.12, 0.35, 0.30, 0.30, 0.02, 0.02], dtype=np.float32)

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
    cos = np.clip(dot / denom, -1.0, 1.0)
    return np.arccos(cos)


def _finger_closure(joints: np.ndarray, ids: list[int]) -> np.ndarray:
    p = joints[:, ids, :]
    b0 = p[:, 1] - p[:, 0]
    b1 = p[:, 2] - p[:, 1]
    b2 = p[:, 3] - p[:, 2]
    a0 = _angle_between(b0, b1)
    a1 = _angle_between(b1, b2)
    return np.clip((0.65 * a0 + 0.35 * a1) / 1.35, 0.0, 1.0)


def _thumb_opposition(joints: np.ndarray) -> np.ndarray:
    wrist = joints[:, 0]
    index_mcp = joints[:, 5]
    pinky_mcp = joints[:, 17]
    thumb_tip = joints[:, 4]
    palm_x = index_mcp - pinky_mcp
    palm_y = 0.5 * (index_mcp + pinky_mcp) - wrist
    palm_normal = np.cross(palm_x, palm_y)
    thumb_vec = thumb_tip - wrist
    thumb_across_palm = np.sum(thumb_vec * palm_x, axis=1) / (_safe_norm(thumb_vec) * _safe_norm(palm_x))
    thumb_into_palm = np.sum(thumb_vec * palm_normal, axis=1) / (_safe_norm(thumb_vec) * _safe_norm(palm_normal))
    raw = 0.65 * ((thumb_across_palm + 1.0) * 0.5) + 0.35 * ((thumb_into_palm + 1.0) * 0.5)
    return np.clip(raw, 0.0, 1.0)


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


def _aero_ctrl_to_qpos(ctrl: np.ndarray) -> np.ndarray:
    qpos = np.zeros((ctrl.shape[0], 11), dtype=np.float32)
    qpos[:, 0] = ctrl[:, 2]
    qpos[:, 1] = 0.925 * ctrl[:, 2]
    qpos[:, 2] = ctrl[:, 3]
    qpos[:, 3] = 0.925 * ctrl[:, 3]
    qpos[:, 4] = np.clip(ctrl[:, 4], 0.0, 1.3963)
    qpos[:, 5] = np.clip(0.925 * ctrl[:, 4], 0.0, 1.2915)
    qpos[:, 6] = np.clip(ctrl[:, 5], 0.0, 1.3963)
    qpos[:, 7] = np.clip(0.925 * ctrl[:, 5], 0.0, 1.2915)
    qpos[:, 8] = np.clip(ctrl[:, 0], 0.0, 1.3788)
    qpos[:, 9] = np.clip(0.16 * ctrl[:, 0], 0.0, 0.5236)
    qpos[:, 10] = np.clip(ctrl[:, 1], 0.0, 0.7854)
    return qpos


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


def _hand_joints_to_aero_ctrl(joints: np.ndarray) -> np.ndarray:
    index = _finger_closure(joints, FINGER_JOINTS["index"])
    middle = _finger_closure(joints, FINGER_JOINTS["middle"])
    ring = _finger_closure(joints, FINGER_JOINTS["ring"])
    pinky = _finger_closure(joints, FINGER_JOINTS["pinky"])
    thumb_flex = _finger_closure(joints, FINGER_JOINTS["thumb"])
    thumb_rot = _thumb_opposition(joints)

    amount = np.stack([thumb_rot, thumb_flex, index, middle, ring, pinky], axis=1)
    ctrl = AERO_CTRL_LOW[None, :] + amount * (AERO_CTRL_HIGH - AERO_CTRL_LOW)[None, :]
    return np.clip(ctrl, AERO_CTRL_LOW[None, :], AERO_CTRL_HIGH[None, :])


def build_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    hand_joints, joint_files = _load_hand_joints(args.hand_dir)
    ctrl = _hand_joints_to_aero_ctrl(hand_joints)
    ctrl = _moving_average(ctrl, args.smoothing_window)
    qpos = _aero_ctrl_to_qpos(ctrl)
    action = np.clip(
        (ctrl - AERO_DEFAULT_CTRL[None, :]) / AERO_ACTION_SCALE[None, :],
        -1.0,
        1.0,
    ).astype(np.float32)
    time = np.arange(ctrl.shape[0], dtype=np.float32) * np.float32(args.dt)
    object_poses = _load_object_pose_dir(args.object_pose_dir, ctrl.shape[0])
    metadata = {
        "source": str(args.hand_dir),
        "target_hand": "Aero-Hand V2 cube grasp hand",
        "dt": float(args.dt),
        "retarget_type": "direct_dexmv_video_to_aero_ctrl",
        "mapping": {
            "thumb_rot": "DexMV 21-joint thumb opposition relative to palm plane",
            "thumb_flex": "DexMV thumb bend angles",
            "index/middle/ring/pinky": "DexMV finger bend angles mapped to Aero coupled MCP controls",
        },
        "joint_files_first_last": [joint_files[0], joint_files[-1]],
    }
    return {
        "dexmv_hand_joints": hand_joints.astype(np.float32),
        "aero_prior_ctrl": ctrl.astype(np.float32),
        "aero_prior_qpos": qpos.astype(np.float32),
        "aero_prior_action": action,
        "aero_default_ctrl": AERO_DEFAULT_CTRL.copy(),
        "aero_action_scale": AERO_ACTION_SCALE.copy(),
        "aero_ctrl_low": AERO_CTRL_LOW.copy(),
        "aero_ctrl_high": AERO_CTRL_HIGH.copy(),
        "time": time,
        "dt": np.array(args.dt, dtype=np.float32),
        "object_poses": object_poses,
        "aero_action_names": np.array(AERO_ACTION_NAMES),
        "aero_qpos_names": np.array(AERO_QPOS_NAMES),
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
    parser.add_argument(
        "--object-pose-dir",
        type=Path,
        default=None,
        help="Optional DexMV object_pose directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ppo_bridge/demo_dataset_aero_direct.npz"),
        help="Output .npz consumed by the Aero direct residual PPO environment.",
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--smoothing-window", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **dataset)
    ctrl = dataset["aero_prior_ctrl"]
    print(f"Wrote {args.output}")
    print(f"frames={ctrl.shape[0]}, aero_prior_ctrl_shape={ctrl.shape}")
    print(f"aero_prior_ctrl_min={np.round(ctrl.min(axis=0), 4).tolist()}")
    print(f"aero_prior_ctrl_max={np.round(ctrl.max(axis=0), 4).tolist()}")


if __name__ == "__main__":
    main()
