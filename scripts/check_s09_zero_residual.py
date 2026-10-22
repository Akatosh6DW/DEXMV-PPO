"""Distinguish nominal control equivalence from bitwise chaotic-physics replay."""

import json
import numpy as np

from s09_suspension import OUT
from retarget_s09_reference import digest


def main():
    paths=[OUT/"diagnostics/reference_development/step_000000000",
           OUT/"train/residual_s9101/step_000000000"]
    rows=[json.loads((p/"episodes.json").read_text()) for p in paths]
    assert [r["seed"] for r in rows[0]]==[r["seed"] for r in rows[1]]
    matched=all(a["success30"]==b["success30"] and abs(a["hold_sec"]-b["hold_sec"])<1e-5
                for a,b in zip(*rows))
    with np.load(paths[0]/"preview_trajectories.npz") as a,np.load(paths[1]/"preview_trajectories.npz") as b:
        np.testing.assert_array_equal(a["seeds"],b["seeds"])
        np.testing.assert_array_equal(b["requested_residual"],0)
        previews=[dict(seed=int(seed),success30=bool(a["success30"][i,-1]),
            max_qpos_component_difference=float(np.max(np.abs(a["qpos"][i]-b["qpos"][i]))),
            max_ctrl_difference_rad=float(np.max(np.abs(a["ctrl"][i]-b["ctrl"][i])))) for i,seed in enumerate(a["seeds"])]
    assert matched and all(r["max_ctrl_difference_rad"]<2e-7 for r in previews)
    result=dict(outcome_match=matched,episode_count=len(rows[0]),control_tolerance_rad=2e-7,
        bitwise_physics_match=all(r["max_qpos_component_difference"]==0 for r in previews),
        previews=previews,source_hashes={str(p/"preview_trajectories.npz"):digest(p/"preview_trajectories.npz") for p in paths},
        limitation="Reference absolute-action roundtrip differs from direct residual targets at float32 rounding. Failed contact trajectories amplify tiny control differences; do not claim bitwise equivalence or generalize eight development outcomes to all test states.")
    (OUT/"diagnostics/zero_residual_equivalence.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
