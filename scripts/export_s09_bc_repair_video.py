"""Export one success and one failure from a predetermined repaired model."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from retarget_s09_reference import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirmation", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.confirmation/"plan.json").read_text())
    job = next(j for j in plan["jobs"] if j["name"] == "new_bc_ppo_s13101")
    folder = ROOT/job["output"]/"step_000000000"
    with np.load(folder/"preview_trajectories.npz", allow_pickle=False) as z:
        seeds, success = z["seeds"], z["success30"][:,-1].astype(bool)
    output = a.confirmation/"videos"
    staging = a.confirmation/"rendered_cases"
    output.mkdir(exist_ok=False)
    all_videos = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", LP_NUM_THREADS="2")
    for status in (True, False):
        indices = np.flatnonzero(success == status)
        if not len(indices):
            continue
        index = int(indices[0])
        seed = int(seeds[index])
        case = staging/f"seed_{seed}"
        label = f"修复BC后PPO_训练13101_测试{seed}"
        command = [sys.executable, str(ROOT/"scripts/render_s09_trace.py"),
            "--model", str(ROOT/plan["input"]/"model.xml"),
            "--trace", str(folder/"preview_trajectories.npz"),
            "--initial", str(folder/"initial_states.npz"),
            "--episode", str(index), "--output", str(case), "--label", label]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        report = json.loads((case/"render.json").read_text())
        for video in report["videos"]:
            shutil.copyfile(case/video["file"], output/video["file"])
            all_videos.append(dict(**video, training_seed=13101, test_seed=seed, success30=status,
                trace_sha256=report["trace_sha256"], cut_reason=report["cut_reason"]))
        shutil.copyfile(case/"contact_sheet.jpg", output/f"测试{seed}_首尾对照.jpg")
    lines = ["# S09修复BC+PPO视频目录", "",
        "只展示预先指定的训练种子13101，不选成绩最高的训练种子。成功/失败各选保存预览中的第一例。",
        "这些案例不是成功率抽样统计；全部128个测试初态和三个训练种子见本实验的保留测试报告。",
        "任务是随机朝向圆柱悬吊3秒后释放，最终要求严格连续合格30秒且回合末仍合格。",
        "视频从初态开始，成功片只保留拿稳4秒，失败片停在首次失去合格抓持/掉落。",
        "外观版看机械手表面；碰撞几何版看物理接触形状。两版来自同一条已审计轨迹。", ""]
    for item in all_videos:
        lines.append(f"- [{item['file']}]({item['file']})：{item['duration_sec']:.2f}秒；完整40秒评估{'成功' if item['success30'] else '失败'}。")
    (output/"目录.md").write_text("\n".join(lines)+"\n")
    (output/"videos.json").write_text(json.dumps(all_videos,ensure_ascii=False,indent=2)+"\n")
    for item in all_videos:
        assert digest(output/item["file"]) == item["sha256"]
    print(json.dumps(dict(output=str(output), videos=len(all_videos)),ensure_ascii=False))


if __name__ == "__main__":
    main()
