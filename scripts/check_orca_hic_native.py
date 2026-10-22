"""Native replay of HIC pinch timing with the geometry-calibrated endpoint."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from diagnose_orca_grasp import SCENE, rollout


source = Path("artifacts/cube20/hic20/prepared/human_cube.npz")
geometry = Path("artifacts/orca30/geometry_o04/report.json")
output = Path("artifacts/orca30/hic_native_o09")
output.mkdir(parents=True, exist_ok=True)
with np.load(source) as z:
    joints = z["hand_joints"][:61]
distance = np.linalg.norm(joints[:, 4] - joints[:, 8], axis=1)
phase = np.maximum.accumulate(np.clip((distance[0] - distance) / (distance[0] - distance.min()), 0, 1))
best = json.loads(geometry.read_text())["best"]
model = mujoco.MjModel.from_xml_path(str(SCENE))
result = rollout(model, best["cube"], np.array(best["ctrl"]), seconds=40,
                 phase=phase, trace=True)
(output / "trace.json").write_text(json.dumps(result))
result.pop("trace")
result.update(kind="native_HIC_timing_geometric_endpoint_not_RL",
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest())
(output / "report.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
