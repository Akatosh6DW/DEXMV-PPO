#!/usr/bin/env python3
"""Refine only the hold endpoint after the unchanged 3D video trajectory."""

import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from diagnose_orca_grasp import SCENE, model_indices, rollout


def main():
    source = Path("artifacts/orca30/dexmv_trajectory_o16/reference.npz")
    output = Path("artifacts/orca30/dexmv_refined_o18")
    output.mkdir(parents=True, exist_ok=True)
    reference = np.load(source)["ctrl"]
    cube = [.055, .025, .265]
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    qids, _, gids = model_indices(model)
    endpoint = reference[-1]
    data.qpos[qids] = endpoint
    mujoco.mj_forward(model, data)
    tip_pos = data.geom_xpos[gids[:3]].copy()
    selected = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    low, high = model.actuator_ctrlrange[selected].T
    records, sequences = [], []
    for x, y, z in itertools.product([-.020, -.015, -.010, -.005, 0], [-.004, 0, .004], [0, .003, .006]):
        shift = np.array([x, y, z])
        target = tip_pos + shift
        def residual(q):
            ctrl = endpoint.copy()
            ctrl[selected] = q
            data.qpos[qids] = ctrl
            mujoco.mj_forward(model, data)
            error = (data.geom_xpos[gids[:3]] - target) * np.array([1, 1, .3])[:, None]
            return np.concatenate([error.ravel(), .001 * (q - endpoint[selected])])
        fit = least_squares(residual, np.clip(endpoint[selected], low + 1e-8, high - 1e-8),
                            bounds=(low, high), max_nfev=100)
        close = endpoint.copy()
        close[selected] = fit.x
        tail = endpoint + np.linspace(.1, 1, 10)[:, None] * (close - endpoint)
        sequence = np.concatenate([reference, tail])
        result = rollout(model, cube, close, seconds=40, control_sequence=sequence)
        result.pop("trace")
        result.update(shift=shift.tolist(), ctrl=close.tolist())
        records.append(result)
        sequences.append(sequence)
        print(json.dumps({k: result[k] for k in ("shift", "hold_sec", "end_sec")}), flush=True)
    index = max(range(len(records)), key=lambda i: (records[i]["hold_sec"], -np.linalg.norm(records[i]["shift"])))
    best = records[index]
    np.savez_compressed(output / "reference.npz", ctrl=sequences[index], time=np.arange(len(sequences[index])) * .05)
    trace = rollout(model, cube, np.array(best["ctrl"]), seconds=40, trace=True, control_sequence=sequences[index])
    (output / "native_trace.json").write_text(json.dumps(trace))
    (output / "report.json").write_text(json.dumps(dict(
        source=str(source), method="unchanged video 3D acquisition + physically refined hold endpoint",
        best=best, selected_index=index, records=records), indent=2))
    print("BEST", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
