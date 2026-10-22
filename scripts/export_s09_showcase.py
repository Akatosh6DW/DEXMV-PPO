"""Principal four-method clips for uniform and web-axis S09 tests only."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from retarget_s09_reference import digest
from run_s09_study import METHODS, OUT, ROOT, SEEDS

DELIVERY=ROOT/"deliverables/S09_多朝向悬吊圆柱_2026-10-15"
LABELS={"ppo":"纯PPO","guided":"BC后PPO","residual":"视频参考加残差PPO","reference":"视频参考回放"}


def main():
    audit=json.loads((OUT/"report/matrix_verification.json").read_text())
    assert audit["status"]=="matrix_verified" and audit["jobs"]==50
    videos=[]
    for orientation in ("uniform","web"):
        for method in (*METHODS,"reference"):
            name="reference" if method=="reference" else f"{method}_s{SEEDS[0]}"
            run=OUT/"eval"/orientation/name
            folder=run/"step_000000000"
            episodes={r["seed"]:r for r in json.loads((folder/"episodes.json").read_text())}
            with np.load(folder/"preview_trajectories.npz") as z:
                seeds=z["seeds"].copy()
            index=next((i for i,seed in enumerate(seeds) if episodes[int(seed)]["success30"]),0)
            seed=int(seeds[index])
            target=DELIVERY/f"{orientation}_{name}_{seed}"
            if not (target/"render.json").exists():
                subprocess.run([sys.executable,str(ROOT/"scripts/render_s09_trace.py"),
                    "--model",str(OUT/"inputs/model.xml"),"--trace",str(folder/"preview_trajectories.npz"),
                    "--initial",str(folder/"initial_states.npz"),"--episode",str(index),
                    "--output",str(target),"--label",f"{'随机朝向' if orientation=='uniform' else '虎口方向'}_{LABELS[method]}_初态{seed}"],check=True,cwd=ROOT)
            rendered=json.loads((target/"render.json").read_text())
            assert rendered["success30"]==episodes[seed]["success30"]
            assert rendered["trace_sha256"] == digest(folder/"preview_trajectories.npz")
            for item in rendered["videos"]:
                file=target/item["file"]
                assert digest(file)==item["sha256"] and item["decoded"]
                videos.append(dict(file=str(file.relative_to(DELIVERY)),orientation=orientation,method=method,
                    seed=seed,training_seed=None if method=="reference" else SEEDS[0],
                    full_episode=episodes[seed],**{k:v for k,v in item.items() if k!="file"}))
    DELIVERY.mkdir(exist_ok=True)
    (DELIVERY/"主要视频索引.json").write_text(json.dumps(videos,ensure_ascii=False,indent=2)+"\n")
    lines=["# S09多朝向悬吊圆柱：主要视频","", "只展示四个正式方法，手心角度，外观/碰撞体各一版；随机朝向与虎口方向各选一个案例。",
        "学习方法固定首个训练种子9101；在已保存的预览中优先选第一条成功，否则第一条失败。案例选择不代表全部测试成功率。",
        "成功片在连续有效抓稳4秒后结束，但其完整评估必须通过30秒并保持到回合结束。失败片截到首次丢失已建立的有效抓持或掉落；若一直没抓住而未终止，则保留到评估末尾。",
        "绳子3秒释放；初始是理想平衡吊带，不是隐藏姿态伺服。下方开发参考文件夹不属于正式PPO成绩。","",
        "| 视频 | 完整回合是否成功 | 最长保持秒 | 视频秒 |","|---|---|---:|---:|"]
    for row in videos:
        e=row["full_episode"]
        lines.append(f"| [{Path(row['file']).name}]({row['file']}) | {e['success30']} | {e['hold_sec']:.2f} | {row['duration_sec']:.2f} |")
    lines += ["","完整数据与报告：主工程`artifacts/orca_benchmark_suite/S09/report/`；方法原理：`docs/experiments/S09/`。"]
    (DELIVERY/"目录.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(videos=len(videos),output=str(DELIVERY)),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
