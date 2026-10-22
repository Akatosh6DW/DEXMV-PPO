"""Palm-view S09 clips; physical string is shown only before its release."""

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LP_NUM_THREADS", "4")

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from inspect_orca_mesh_alignment import build_overlay
from retarget_s09_reference import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--trace",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--label",required=True)
    p.add_argument("--episode",type=int)
    p.add_argument("--initial",type=Path)
    p.add_argument("--reference",type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.trace) as z:
        trace={k:(v[a.episode] if a.episode is not None and k!="seeds" else v) for k,v in z.items()}
    if trace["qpos"].ndim!=2:
        raise ValueError("Select a saved episode, not an entire batch")
    hold=trace.get("hold",trace.get("hold_sec"))
    alive=trace["alive"].astype(bool)
    valid=trace.get("valid",hold>0)
    success=bool(trace["success30"][-1])
    if success:
        end=int(np.flatnonzero(hold>=4)[0])
        reason="successful full evaluation; clip stops after four seconds of valid unsupported hold"
    else:
        lost=np.flatnonzero((np.arange(len(hold))>60)&(np.maximum.accumulate(hold)>0)&~valid)
        dropped=np.flatnonzero(~alive)
        options=[int(x[0]) for x in (lost,dropped) if len(x)]
        end=min(options) if options else len(hold)-1
        reason="first loss of valid released grasp/drop; full-horizon failure status retained"
    model,overlay=build_overlay(a.model,a.output/"visual_overlay.xml")
    model.vis.global_.offwidth,model.vis.global_.offheight=960,720
    data=mujoco.MjData(model)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[.08,.055,.275]
    camera.distance,camera.azimuth,camera.elevation=.48,-105,-12
    initial=None
    if a.initial:
        with np.load(a.initial) as z:
            initial=z["qpos"]
            if initial.ndim==2:
                if "seeds" in trace and "seeds" in z:
                    seed=trace["seeds"][a.episode]
                    initial=initial[int(np.flatnonzero(z["seeds"]==seed)[0])]
                else:
                    initial=initial[a.episode or 0]
    elif a.reference:
        from diagnose_orca_grasp import model_indices
        qids,oid,_=model_indices(model)
        with np.load(a.reference) as z:
            initial=model.qpos0.copy()
            initial[qids]=z["initial_ctrl"]
            initial[oid:oid+7]=np.r_[z["object_position"],z["object_quaternion"]]
    paths=[]
    panels=[]
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
                t=0 if index<0 else (i+1)*.05
                # Do not draw an intact string after the simulated release.
                model.tendon_rgba[:,3]=float(t<3.)
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera,scene_option=option)
                picture=Image.fromarray(renderer.render())
                draw=ImageDraw.Draw(picture)
                draw.rectangle((0,0,960,42),fill=(24,26,28))
                draw.text((10,12),f"S09 suspended cylinder | t={t:.2f}s | hold={0 if index<0 else hold[i]:.2f}s | rope={'on' if t<3 else 'released'}",fill="white")
                frames.append(np.asarray(picture))
                if index==indices[0] or index==end:
                    picture.save(a.output/f"{mode}_{max(index,0):04d}.png")
                    panels.append(picture.resize((480,360)))
            path=a.output/f"圆柱_S09_{a.label}_手心_{'成功' if success else '失败'}_{mode}.mp4"
            mediapy.write_video(str(path),frames,fps=20)
            subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-f","null","-"],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            paths.append(dict(file=path.name,sha256=digest(path),decoded=True,duration_sec=len(frames)/20))
    sheet=Image.new("RGB",(960,720))
    for i,picture in enumerate(panels):
        sheet.paste(picture,((i%2)*480,(i//2)*360))
    sheet.save(a.output/"contact_sheet.jpg")
    report=dict(label=a.label,trace=str(a.trace),trace_sha256=digest(a.trace),success30=success,
                clip_end_frame=end,cut_reason=reason,overlay=overlay,videos=paths)
    (a.output/"render.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    lines=[f"# S09 {a.label}","",f"完整评估是否通过30秒：{success}。视频只展示指定案例，不代表总体成功率。",
           "悬吊为理想平衡吊带，3秒释放后不再显示完整绳索；参考开发片不冒充PPO训练成果。","",
           *[f"- [{r['file']}]({r['file']})" for r in paths]]
    (a.output/"目录.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
