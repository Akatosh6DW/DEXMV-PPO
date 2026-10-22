"""Render saved S10 states, showing actual hand approach and grasp release."""

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("PYOPENGL_PLATFORM","egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE","1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES","/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LP_NUM_THREADS","4")

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from inspect_orca_mesh_alignment import build_overlay
from s10_mobile_base import CENTER, OUT, rotation
from retarget_s09_reference import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--label",required=True)
    p.add_argument("--episode",type=int)
    p.add_argument("--initial",type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.trace) as z:
        trace={k:(v[a.episode] if a.episode is not None and k!="seeds" else v) for k,v in z.items()}
    if trace["qpos"].ndim!=2:
        raise ValueError("Choose one saved episode")
    hold=trace.get("hold",trace.get("hold_sec"))
    success=bool(trace["success30"][-1])
    if success:
        end=int(np.flatnonzero(hold>=4)[0])
    else:
        lost=np.flatnonzero((np.arange(len(hold))>=100)&(np.maximum.accumulate(hold)>0)&(hold==0))
        dropped=np.flatnonzero(~trace["alive"].astype(bool))
        ends=[int(v[0]) for v in (lost,dropped) if len(v)]
        end=min(ends) if ends else len(hold)-1
    model,overlay=build_overlay(OUT/"inputs/model.xml",a.output/"visual_overlay.xml")
    model.vis.global_.offwidth,model.vis.global_.offheight=960,720
    model.vis.headlight.ambient[:]=.45
    model.vis.headlight.diffuse[:]=.65
    model.vis.headlight.specular[:]=.2
    data=mujoco.MjData(model)
    rid=int(model.joint("s10_hand_free").qposadr[0])
    mount=model.body("right_mount").id
    final_index=min(100,len(trace["qpos"])-1)
    transform=rotation(trace["qpos"][final_index,rid+3:rid+7])*rotation(model.body_quat[mount]).inv()
    az,el=np.deg2rad([-105.,-12.])
    direction=transform.apply([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
    camera=mujoco.MjvCamera()
    camera.lookat[:]=CENTER+[0.,-.025,-.025]
    camera.distance=.65
    camera.azimuth=np.rad2deg(np.arctan2(direction[1],direction[0]))
    camera.elevation=np.rad2deg(np.arcsin(np.clip(direction[2],-1,1)))
    initial=None
    if a.initial:
        with np.load(a.initial) as z:
            seed=int(trace["seeds"][a.episode])
            initial=z["qpos"][int(np.flatnonzero(z["seeds"]==seed)[0])]
    # Fit the entire approach, including the forearm CAD, not only the final grasp.
    visible=np.array([(model.geom(i).name or "").startswith("cad_visual_") or
                      model.geom(i).name=="hic20_cube_visual" for i in range(model.ngeom)])
    centers=[]
    for index in [-1,*range(0,end+1,10),end]:
        data.qpos[:]=initial if index<0 and initial is not None else trace["qpos"][max(index,0)]
        mujoco.mj_forward(model,data)
        centers.append(data.geom_xpos[visible].copy())
    points=np.concatenate(centers)
    radii=np.tile(model.geom_rbound[visible],len(centers))
    camera.lookat[:]=.5*((points-radii[:,None]).min(0)+(points+radii[:,None]).max(0))
    right=np.cross([0.,0.,1.],direction); right/=np.linalg.norm(right)
    up=np.cross(direction,right)
    relative=points-camera.lookat
    tangent=np.tan(np.deg2rad(model.vis.global_.fovy)*.5)
    span=np.maximum((np.abs(relative@right)+radii)/(960/720),np.abs(relative@up)+radii)
    camera.distance=float(np.max(relative@direction+span/tangent)*1.08)
    paths=[]
    with mujoco.Renderer(model,720,960) as renderer:
        for mode in ("外观","碰撞几何"):
            option=mujoco.MjvOption()
            option.geomgroup[2],option.geomgroup[3]=mode=="外观",mode=="碰撞几何"
            frames=[]
            indices=([-1] if initial is not None else [])+list(range(end+1))
            for index in indices:
                i=max(index,0)
                data.qpos[:]=initial if index<0 else trace["qpos"][i]
                data.qvel[:]=0 if index<0 else trace["qvel"][i]
                data.ctrl[:]=0 if index<0 else trace["ctrl"][i]
                data.mocap_pos[:]=trace["mocap"][i]
                data.mocap_quat[:]=trace["mocap_quat"][i]
                t=0 if index<0 else (i+1)*.05
                model.tendon_rgba[:,3]=float(t<5.)
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera,scene_option=option)
                picture=Image.fromarray(renderer.render())
                draw=ImageDraw.Draw(picture)
                draw.rectangle((0,0,960,42),fill=(24,26,28))
                draw.text((10,12),f"S10 moving hand | t={t:.2f}s | hold={0 if index<0 else hold[i]:.2f}s | rope={'on' if t<5 else 'released'} | full test={'PASS' if success else 'FAIL'}",fill="white")
                frame=np.asarray(picture)
                if frame.std()<5:
                    raise ValueError("Blank rendered frame")
                frames.append(frame)
                if index in (indices[0],39,end):
                    picture.save(a.output/f"{mode}_{max(0,index):04d}.png")
            path=a.output/f"圆柱_S10_整手对准_{a.label}_手心_{'成功' if success else '失败'}_{mode}.mp4"
            mediapy.write_video(str(path),frames,fps=20)
            subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-f","null","-"],
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            paths.append(dict(file=path.name,sha256=digest(path),duration_sec=len(frames)/20,decoded=True))
    report=dict(trace=str(a.trace.resolve()),trace_sha256=digest(a.trace),label=a.label,success30=success,
                clip_end_frame=end,overlay=overlay,videos=paths,
                cut="First four seconds of valid released hold, or first loss/drop for failed full evaluation")
    (a.output/"render.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
