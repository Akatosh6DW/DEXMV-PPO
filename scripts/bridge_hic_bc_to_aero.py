#!/usr/bin/env python3
"""Bridge the learned HIC/DexMV BC prior, preserving physical timestamps."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bridge_dexmv_to_aero_demo import _adroit_to_aero_ctrl, _aero_ctrl_to_qpos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("artifacts/cube20/hic20/dexmv_bc/learned_prior.npz"))
    p.add_argument("--retarget", type=Path, default=Path("artifacts/cube20/hic20/dexmv_retarget/adroit_cube.npz"))
    p.add_argument("--output", type=Path, default=Path("artifacts/cube20/hic20/ppo_bridge"))
    p.add_argument("--blend", type=float, default=0.1)
    a = p.parse_args()
    learned, ref = np.load(a.input), np.load(a.retarget)
    tips = ref["fitted_positions"][:, [3, 6]]
    obj = ref["object_poses"]
    local = np.einsum("tji,tkj->tki", obj[:, :3, :3], tips - obj[:, None, :3, 3])
    distances = np.linalg.norm(np.maximum(np.abs(local) - .0265, 0), axis=-1)
    # Stop at the best thumb/index approach, before the human places/releases the cube.
    stop_frame = int(np.argmin(distances.max(axis=1)))
    stop_time = float(ref["time"][stop_frame])
    keep = learned["time"] <= stop_time
    qpos = learned["adroit_qpos"][keep]
    ctrl = _adroit_to_aero_ctrl(qpos)
    meta = {"source": str(a.input.resolve()), "source_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
            "bc_report": json.loads(learned["metadata_json"].item()),
            "mapping": "existing Adroit 30-DoF to Aero 6-actuator flexion mapping",
            "time_resampling": "BC model evaluated at physical 0.05 second intervals",
            "stop_source_video_frame": stop_frame, "stop_source_time_sec": stop_time,
            "selection": "minimum maximum thumb/index tip distance to cube surface; freeze thereafter",
            "source_cube_size_m": 0.053, "target_cube_pose": "unchanged Aero cube task reset",
            "root_motion_transferred": False}
    a.output.mkdir(parents=True, exist_ok=True)
    dataset = a.output / "demo_dataset.npz"
    np.savez_compressed(dataset, aero_prior_ctrl=ctrl, aero_prior_qpos=_aero_ctrl_to_qpos(ctrl),
                        adroit_qpos=qpos, time=learned["time"][keep], dt=learned["dt"],
                        metadata_json=np.array(json.dumps(meta)))
    config = {"residual_prior": {"enabled": True, "dataset_path": str(dataset.resolve()),
              "ctrl_key": "aero_prior_ctrl", "blend_alpha": a.blend, "residual_scale": 1.0, "loop": False}}
    (a.output / "env_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (a.output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"dataset": str(dataset), "frames": len(ctrl), "stop_video_frame": stop_frame,
                      "ctrl_min": ctrl.min(0).tolist(), "ctrl_max": ctrl.max(0).tolist()}, indent=2))


if __name__ == "__main__":
    main()
