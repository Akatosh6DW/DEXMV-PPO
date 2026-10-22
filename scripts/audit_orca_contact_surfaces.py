"""Unsigned STL distance at actual saved-state contact midpoints, not volume IoU."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from fit_orca_collision import distance_to_mesh, load_parts, mesh_scene


def audit(model_path,trace_path,stride):
    model=mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data=mujoco.MjData(model)
    cube=model.geom('cube').id
    scenes={body:mesh_scene(trimesh.util.concatenate([mesh for _,mesh in parts]))
            for body,parts in load_parts().items()}
    points={};depths={}
    with np.load(trace_path) as z:
        for episode in range(len(z['qpos'])):
            for step in range(0,z['qpos'].shape[1],stride):
                if not z['active'][episode,step] or not z['released'][episode,step]:
                    continue
                data.qpos[:]=z['qpos'][episode,step]
                data.qvel[:]=z['qvel'][episode,step]
                data.ctrl[:]=z['ctrl'][episode,step]
                data.mocap_pos[:]=z['mocap'][episode,step]
                mujoco.mj_forward(model,data)
                for c in data.contact:
                    if c.dist>0 or cube not in c.geom:
                        continue
                    hand=int(c.geom[1] if c.geom[0]==cube else c.geom[0])
                    bid=int(model.geom_bodyid[hand])
                    body=model.body(bid).name
                    if body not in scenes:
                        continue
                    local=(c.pos-data.xpos[bid])@data.xmat[bid].reshape(3,3)
                    points.setdefault(body,[]).append(local.copy())
                    depths.setdefault(body,[]).append(-float(c.dist)*1000)
    rows={}
    all_distances=[]
    for body,p in points.items():
        distances=distance_to_mesh(scenes[body],p)*1000
        all_distances.extend(distances.tolist())
        rows[body]=dict(samples=len(p),midpoint_stl_p50_mm=float(np.median(distances)),
                        midpoint_stl_p95_mm=float(np.quantile(distances,.95)),
                        midpoint_stl_max_mm=float(max(distances)),
                        penetration_p95_mm=float(np.quantile(depths[body],.95)))
    return dict(model=str(model_path),trace=str(trace_path),stride=stride,bodies=rows,
                all_midpoint_stl_p95_mm=float(np.quantile(all_distances,.95)) if all_distances else None,
                note='Native forward on saved simulation states; geometric contact midpoint to original STL. '
                     'Not an exact proxy-surface distance: midpoint includes soft-contact penetration. '
                     'Active released states only, including the terminal transition; no resimulation or success rescoring.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--trace',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--stride',type=int,default=20)
    args=p.parse_args()
    if args.stride<1:
        p.error('Stride must be positive')
    result=audit(args.model,args.trace,args.stride)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
