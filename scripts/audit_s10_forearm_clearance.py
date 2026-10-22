"""Check visual-only forearm CAD hulls against the cylinder in saved previews.

Positive convex-hull clearance is conservative for the underlying mesh.
Negative hull distance requires inspection; it does not alone prove a concave
CAD surface intersection. This script never changes simulated collision masks.
"""

import argparse
import json

import mujoco
import numpy as np

from inspect_orca_mesh_alignment import build_overlay
from s10_mobile_base import OUT
from retarget_s09_reference import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name",required=True)
    a=p.parse_args()
    out=OUT/"audits"/a.name
    out.mkdir(parents=True,exist_ok=False)
    model,_=build_overlay(OUT/"inputs/model.xml",out/"overlay.xml")
    data=mujoco.MjData(model)
    geoms=[model.geom(f"cad_visual_{i}").id for i in range(3)]
    expected={"right_ForeArmStructure-Model_e18f2368","right_TopTower-Model_4a80d30e"}
    assert all(model.body(model.geom_bodyid[g]).name in expected for g in geoms)
    cube=model.geom("cube").id
    rows=[]
    for trace in sorted((OUT/"eval").glob("*/step_000000000/preview_trajectories.npz")):
        with np.load(trace) as z:
            for index,seed in enumerate(z["seeds"]):
                minimum=1.; closest=None
                for step in np.flatnonzero(z["active"][index]):
                    data.qpos[:]=z["qpos"][index,step]
                    mujoco.mj_kinematics(model,data)
                    for geom in geoms:
                        distance=mujoco.mj_geomDistance(model,data,int(geom),cube,1.,None)
                        if distance<minimum:
                            minimum=float(distance); closest=dict(step=int(step),geom=model.geom(geom).name)
                rows.append(dict(run=trace.parent.parent.name,seed=int(seed),
                    success30=bool(z["success30"][index,-1]),min_forearm_hull_clearance_m=minimum,
                    closest=closest,trace_sha256=digest(trace)))
    report=dict(status="clear_in_checked_frames" if all(r["min_forearm_hull_clearance_m"]>0 for r in rows)
                else "intersection_requires_review",episodes=rows,
                scope="All active 20Hz saved preview frames; three upstream forearm/tower CAD convex hulls; not unsaved cases or every 5ms substep")
    (out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
