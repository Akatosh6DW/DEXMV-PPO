"""Put a small, explicitly selected S10 demonstration set in one folder."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np

from s10_mobile_base import OUT, ROOT
from retarget_s09_reference import digest


def main():
    destination=OUT/"videos"
    destination.mkdir(parents=True,exist_ok=True)
    origin=22101
    labels=dict(reference_exact_quality="参考回放_零修正",ppo="从零手指PPO",guided="BC后PPO",residual="参考加残差PPO")
    rows=[]
    for method,label in labels.items():
        source_origin=0 if method.startswith("reference") else origin
        run=OUT/"eval"/f"{method}_s{source_origin}"
        folder=run/"step_000000000"
        trace=folder/"preview_trajectories.npz"
        with np.load(trace) as z:
            seeds=z["seeds"].copy()
            successes=z["success30"][:,-1].astype(bool)
        for success in (True,False):
            indices=np.flatnonzero(successes==success)
            if not len(indices):
                continue
            index=int(indices[0]); seed=int(seeds[index])
            out=OUT/"renders"/f"{method}_s{source_origin}_case{seed}"
            if not (out/"render.json").exists():
                command=[sys.executable,str(ROOT/"scripts/render_s10_trace.py"),"--trace",str(trace),
                    "--initial",str(folder/"initial_states.npz"),"--episode",str(index),"--output",str(out),
                    "--label",f"{label}_初态{seed}"]
                subprocess.run(command,check=True,env=dict(os.environ,MPLCONFIGDIR="/tmp/s10_matplotlib"))
            meta=json.loads((out/"render.json").read_text())
            assert meta["trace_sha256"]==digest(trace) and meta["success30"]==success
            for video in meta["videos"]:
                source=out/video["file"]
                assert digest(source)==video["sha256"]
                shutil.copyfile(source,destination/source.name)
                rows.append(dict(method=method,training_seed=source_origin,
                    case_seed=seed,success30=success,**video,trace=str(trace.relative_to(ROOT))))
    (destination/"manifest.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")
    lines=["# S10 圆柱抓取视频目录","","这里的手会实际从远处接近、转向，再闭合抓取。所有视频集中在此目录，可整体移动。",
        "","采用预先固定的训练来源22101，不挑三个训练来源中最好的。每种方法从保存预览中取第一个成功及第一个失败（若存在）；这是案例演示，不代表总体成功率。",
        "","成功指完整42秒评估通过严格连续30秒抓持，片段只播放到释放后合格保持4秒。失败片段截止首次失去合格抓持/掉落；若始终未建立合格抓持且未掉落，则保留至回合结束。",
        "","第5秒撤去圆柱支撑。外观为CAD网格显示，碰撞几何版为实际接触代理。基座为理想六自由度位姿控制器，不是真实机械臂。",
        "","## 文件","","| 方法 | 初态 | 完整评估 | 时长（秒） | 文件 |","|---|---:|---|---:|---|"]
    for row in rows:
        lines.append(f"| {labels[row['method']]} | {row['case_seed']} | {'成功' if row['success30'] else '失败'} | {row['duration_sec']:.2f} | [{row['file']}]({row['file']}) |")
    lines += ["","原始结果在上一级 `report/results.md`；移动本视频目录后，视频与本目录索引仍可独立使用。"]
    (destination/"目录.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(files=len(rows),directory=str(destination)),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
