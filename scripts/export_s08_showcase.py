"""Export only principal S08 palm-view cases, with honest geometry-reference labels."""

import json
import shutil
import subprocess
import sys

import numpy as np

from prepare_cylinder_s02_source import digest
from run_s08_study import OUT, ROOT, SEEDS
from run_s07_study import write_json

DELIVERY=ROOT/"deliverables/S08_纵向包覆完整对比_2026-10-12"
LABELS={"ppo":"纯PPO","guided":"BC加PPO","residual":"几何参考加残差PPO",
        "bc":"仅BC","reference":"仅几何参考回放"}


def main():
    verified=json.loads((OUT/"report/verification.json").read_text())
    assert verified["status"]=="verified" and verified["jobs"]==86
    DELIVERY.mkdir(exist_ok=False)
    cases=[]
    for method in LABELS:
        name="reference" if method=="reference" else f"{method}_s{SEEDS[0]}"
        run=OUT/"eval/B0_nominal"/name
        episodes=json.loads((run/"step_000000000/episodes.json").read_text())
        success=next((i for i,e in enumerate(episodes) if e["success30"]),None)
        failure=next((i for i,e in enumerate(episodes) if not e["success30"]),None)
        indices=[success if success is not None else failure]
        if method=="residual" and success is not None and failure is not None:
            indices.append(failure)
        cases.extend((method,name,i,episodes[i]) for i in indices)
    videos=[]
    for method,name,i,episode in cases:
        directory=OUT/"renders"/f"{name}_{episode['seed']}"
        if not (directory/"render.json").exists():
            subprocess.run([sys.executable,str(ROOT/"scripts/render_s05_clips.py"),
                "--run",str(OUT/"eval/B0_nominal"/name),"--index",str(i),"--output",str(directory),
                "--task-label",f"圆柱_S08_纵向包覆_初态{episode['seed']}","--header-label","S08 longitudinal wrap",
                "--policy-label",LABELS[method],"--distance",".40"],cwd=ROOT,check=True)
        render=json.loads((directory/"render.json").read_text())
        frame=render["endpoint"]
        with np.load(OUT/"eval/B0_nominal"/name/"step_000000000/preview_trajectories.npz") as z:
            diagnostic=dict(height_loss_mm=float(z["spawn_z"][i,frame]-z["cube_z"][i,frame])*1000,
                hold_sec=float(z["hold"][i,frame]),alive=bool(z["alive"][i,frame]),
                wrap_loads_n=z["wrap_loads"][i,frame].tolist(),
                penetration_mm=float(z["substep_penetration"][i,frame])*1000)
        reason=render["reason"].replace("指腹抓持","包覆抓持")
        if not episode["success30"] and diagnostic["height_loss_mm"]>=15:
            reason+=f"；高度下降{diagnostic['height_loss_mm']:.2f} mm，超过15 mm合格上限"
        for source in sorted(directory.glob("*.mp4")):
            target=DELIVERY/source.name
            shutil.copy2(source,target)
            duration=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration",
                "-of","default=nw=1:nk=1",str(target)],text=True))
            subprocess.run(["ffmpeg","-v","error","-i",str(target),"-f","null","-"],check=True,
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            videos.append(dict(file=target.name,sha256=digest(target),run=name,episode=episode,
                duration_sec=duration,cut_reason=reason,clip_end_diagnostic=diagnostic,decoded=True))
        shutil.copy2(directory/"contact_sheet.jpg",DELIVERY/f"{name}_{episode['seed']}_contact_sheet.jpg")
    write_json(DELIVERY/"来源与校验.json",dict(selection_sha256=digest(OUT/"selection.json"),videos=videos))
    lines=["# S08纵向包覆：主要策略手心视频","",
        "各方法固定首个训练种子8501，展示第一个成功回合；若无成功，则展示第一个失败。残差另补第一条失败（如存在）。这是展示选择，不代替完整测试成功率。",
        "","参考来自几何求解与摆位修复，不是新视频提取；BC+PPO与残差已实际重新训练。",
        "","成功片从起手到连续拿稳4秒；失败片在终止、撤托后接触不合格、或首次丢失已建立有效抓持时结束。若始终未合格但也没有提前终止，则保留到40秒并标记失败。",
        "","物体仍在手中也可能失败，例如高度下降超过15 mm或合格掌侧承力中断。目录明确给出裁切原因，不能只凭是否落地判断任务成功。",
        "","| 视频 | 完整40秒评估是否通过 | 最长合格保持 | 剪辑时长 | 裁切原因 |","|---|---|---:|---:|---|"]
    for v in videos:
        lines.append(f"| [{v['file']}]({v['file']}) | {'是' if v['episode']['success30'] else '否'} | {v['episode']['pad_hold_sec']:.2f}s | {v['duration_sec']:.2f}s | {v['cut_reason']} |")
    lines += ["","完整实验说明见主工程`docs/experiments/S08/完整对比结果.md`；本视频目录不含权重。"]
    (DELIVERY/"目录.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(videos=len(videos),output=str(DELIVERY)),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
