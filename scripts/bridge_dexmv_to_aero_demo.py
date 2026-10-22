#!/usr/bin/env python3
"""Bridge DexMV retargeted Adroit qpos into an Aero-Hand PPO prior dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ADROIT_JOINT_NAMES = [
    "ARTx",
    "ARTy",
    "ARTz",
    "ARRx",
    "ARRy",
    "ARRz",
    "WRJ1",
    "WRJ0",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "FFJ0",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "MFJ0",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "RFJ0",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "LFJ0",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
    "THJ0",
]

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


def _norm(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((x - low) / (high - low), 0.0, 1.0)


def _weighted_flex(qpos: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    idx = list(indices)
    weights = np.array([0.55, 0.30, 0.15], dtype=np.float32)
    flex = np.stack([_norm(qpos[:, i], 0.0, 1.571) for i in idx], axis=1)
    return np.sum(flex * weights[None, :], axis=1)


def _smooth_sequence(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        raise ValueError("--smoothing-window must be odd so the sequence stays centered.")
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.empty_like(values)
    for dim in range(values.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed


def _adroit_to_aero_ctrl(adroit_qpos: np.ndarray) -> np.ndarray:
    """Converts 30D DexMV Adroit qpos into 6D Aero-Hand actuator targets."""
    index = _weighted_flex(adroit_qpos, [9, 10, 11])
    middle = _weighted_flex(adroit_qpos, [13, 14, 15])
    ring = _weighted_flex(adroit_qpos, [17, 18, 19])
    pinky = _weighted_flex(adroit_qpos, [22, 23, 24])

    thumb_rot = _norm(adroit_qpos[:, 25], -1.047, 1.047)
    thumb_flex = (
        0.60 * _norm(adroit_qpos[:, 26], 0.0, 1.309)
        + 0.25 * _norm(adroit_qpos[:, 28], -0.524, 0.524)
        + 0.15 * np.clip(-adroit_qpos[:, 29] / 1.571, 0.0, 1.0)
    )
    amount = np.stack([thumb_rot, thumb_flex, index, middle, ring, pinky], axis=1)
    ctrl = AERO_CTRL_LOW[None, :] + amount * (AERO_CTRL_HIGH - AERO_CTRL_LOW)[None, :]
    return np.clip(ctrl, AERO_CTRL_LOW[None, :], AERO_CTRL_HIGH[None, :])


def _aero_ctrl_to_qpos(ctrl: np.ndarray) -> np.ndarray:
    """Expands Aero 6D actuator targets into the 11D coupled V2 hand qpos."""
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


def _load_object_pose_dir(path: Path | None) -> np.ndarray:
    if path is None:
        return np.zeros((0, 4, 4), dtype=np.float32)
    if not path.exists():
        raise FileNotFoundError(f"Object pose directory not found: {path}")
    poses = []
    for item in sorted(path.glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        pose = np.load(item, allow_pickle=True)
        if pose.shape == () and pose.dtype == object:
            pose_dict = pose.item()
            if not pose_dict:
                continue
            first_key = sorted(pose_dict.keys())[0]
            pose = pose_dict[first_key]
        poses.append(np.asarray(pose, dtype=np.float32))
    if not poses:
        return np.zeros((0, 3, 4), dtype=np.float32)
    return np.stack(poses, axis=0)


def build_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    adroit_qpos = np.load(args.retarget_npy).astype(np.float32)
    if adroit_qpos.ndim != 2 or adroit_qpos.shape[1] != 30:
        raise ValueError(
            f"Expected retarget qpos shape (T, 30), got {adroit_qpos.shape}."
        )

    aero_prior_ctrl = _adroit_to_aero_ctrl(adroit_qpos)
    aero_prior_ctrl = _smooth_sequence(aero_prior_ctrl, args.smoothing_window)
    aero_prior_qpos = _aero_ctrl_to_qpos(aero_prior_ctrl)
    aero_prior_action = np.clip(
        (aero_prior_ctrl - AERO_DEFAULT_CTRL[None, :]) / AERO_ACTION_SCALE[None, :],
        -1.0,
        1.0,
    ).astype(np.float32)

    dt = np.array(args.dt, dtype=np.float32)
    time = np.arange(adroit_qpos.shape[0], dtype=np.float32) * dt
    object_poses = _load_object_pose_dir(args.object_pose_dir)
    if object_poses.shape[0] and object_poses.shape[0] != adroit_qpos.shape[0]:
        target_len = adroit_qpos.shape[0]
        if object_poses.shape[0] > target_len:
            object_poses = object_poses[:target_len]
        else:
            pad_len = target_len - object_poses.shape[0]
            pad = np.repeat(object_poses[-1:, ...], pad_len, axis=0)
            object_poses = np.concatenate([object_poses, pad], axis=0)

    metadata = {
        "source": str(args.retarget_npy),
        "object_pose_dir": str(args.object_pose_dir) if args.object_pose_dir else "",
        "dt": float(args.dt),
        "bridge_version": 1,
        "mapping": {
            "thumb_rot": "Adroit THJ4 normalized from [-1.047, 1.047]",
            "thumb_flex": "weighted Adroit THJ3, THJ1, and THJ0 flexion",
            "index": "weighted Adroit FFJ2/FFJ1/FFJ0 flexion",
            "middle": "weighted Adroit MFJ2/MFJ1/MFJ0 flexion",
            "ring": "weighted Adroit RFJ2/RFJ1/RFJ0 flexion",
            "pinky": "weighted Adroit LFJ2/LFJ1/LFJ0 flexion",
        },
        "aero_action_names": AERO_ACTION_NAMES,
        "aero_qpos_names": AERO_QPOS_NAMES,
        "adroit_joint_names": ADROIT_JOINT_NAMES,
    }

    return {
        "adroit_qpos": adroit_qpos,
        "aero_prior_ctrl": aero_prior_ctrl.astype(np.float32),
        "aero_prior_qpos": aero_prior_qpos.astype(np.float32),
        "aero_prior_action": aero_prior_action,
        "aero_default_ctrl": AERO_DEFAULT_CTRL.copy(),
        "aero_action_scale": AERO_ACTION_SCALE.copy(),
        "aero_ctrl_low": AERO_CTRL_LOW.copy(),
        "aero_ctrl_high": AERO_CTRL_HIGH.copy(),
        "time": time,
        "dt": dt,
        "object_poses": object_poses,
        "adroit_joint_names": np.array(ADROIT_JOINT_NAMES),
        "aero_action_names": np.array(AERO_ACTION_NAMES),
        "aero_qpos_names": np.array(AERO_QPOS_NAMES),
        "metadata_json": np.array(json.dumps(metadata, indent=2), dtype=np.str_),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retarget-npy",
        type=Path,
        default=Path("artifacts/dexmv_retarget/example_retargeting.npy"),
        help="DexMV retarget_human_hand output saved as a (T, 30) .npy file.",
    )
    parser.add_argument(
        "--object-pose-dir",
        type=Path,
        default=None,
        help="Optional DexMV object_pose directory; saved for later reward/task alignment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ppo_bridge/demo_dataset.npz"),
        help="Output .npz read by Aero-Hand residual PPO.",
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Prior frame interval in seconds.")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=3,
        help="Odd moving-average window applied to the 6D Aero prior ctrl.",
    )
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
