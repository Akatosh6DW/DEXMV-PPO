"""Export a flat, provenance-indexed palm-video set for the preselected S12 case."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"ppo":("pure_ppo","纯 PPO"), "bc_initial":("bc_initial","BC 起点"),
    "guided":("bc_plus_ppo","BC 后 PPO"), "reference":("reference_replay","参考回放"),
    "residual":("residual_ppo","参考加残差 PPO")}
REASONS = {
    "four_seconds_stable_hold; full_episode_verified_separately":"完整回合通过，短片到拿稳四秒",
    "drop":"检测到掉落",
    "quality_violation":"首次撤台后接触/形变质量违规",
    "element_collapse_or_inversion":"局部单元塌陷/翻转，不代表真实材料破坏",
    "insufficient_time_remaining_for_15s_hold":"剩余回合时间已不足以完成十五秒连续保持",
    "requested_duration":"保存到指定时长"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    plan = json.loads((a.study/"protocol.json").read_text())
    evaluations = json.loads((a.study/"evaluation_plan.json").read_text())
    if evaluations["protocol_sha256"] != digest(a.study/"protocol.json"):
        raise ValueError("Changed evaluation protocol")
    seed, train_seed = plan["conditions"]["nominal"]["seeds"][0],plan["train_seeds"][0]
    jobs = {j["label"]:j for j in evaluations["jobs"]
            if j["condition"]=="nominal" and j["train_seed"] in (None,train_seed)}
    if set(jobs)!=set(LABELS):
        raise ValueError("Missing preselected controllers")
    for job in jobs.values():
        run = Path(job["output"])
        summary = json.loads((run/"summary.json").read_text())
        if summary["status"]!="completed" or summary["count"]!=job["eval_count"]:
            raise ValueError("Incomplete nominal evaluation")
        config = json.loads((run/"evaluation_config.json").read_text())
        if config["condition"]!="nominal" or config["checkpoint_sha256"]!=job["checkpoint_sha256"]:
            raise ValueError("Video source does not match bound evaluation")
    a.output.mkdir(parents=True,exist_ok=False)
    work = a.study/"video_renders"
    work.mkdir(exist_ok=False)
    env = dict(os.environ,MUJOCO_GL="egl",OPENBLAS_NUM_THREADS="1")
    manifest = []
    for method,(filename,label) in LABELS.items():
        job = jobs[method]
        run = Path(job["output"])
        trace = run/f"trajectory_{seed}.npz"
        episode = next(r for r in json.loads((run/"episodes.json").read_text()) if r["seed"]==seed)
        source = "fixed" if job["train_seed"] is None else f"s{train_seed}"
        for mode in ("visual","collision"):
            folder = work/f"{method}_{mode}"
            subprocess.run([sys.executable,str(ROOT/"scripts/render_s12_soft.py"),str(trace),
                "--model",str(run/"model.xml"),"--output",str(folder),"--short","--mode",mode,
                "--label",f"{filename}_{source}_i{seed}"],cwd=ROOT,env=env,check=True)
            meta = json.loads((folder/"render.json").read_text())
            if meta["full_episode_success"]!=episode["success15"]:
                raise ValueError("Video and episode success disagree")
            video = folder/meta["video"]
            subprocess.run(["ffmpeg","-v","error","-i",str(video),"-f","null","-"],check=True)
            shutil.copy2(video,a.output/video.name)
            shutil.copy2(folder/"render.json",a.output/f"{method}_{mode}_render.json")
            manifest.append(dict(method=method,label=label,train_seed=job["train_seed"],
                initial_seed=seed,mode=mode,file=video.name,sha256=digest(video),
                trace=str(trace.resolve()),trace_sha256=digest(trace),
                checkpoint_sha256=job["checkpoint_sha256"],
                strict_success=episode["success15"],physical_success=episode["physical_success15"],
                full_duration_sec=episode["duration_sec"],best_strict_hold_sec=episode["best_hold_sec"],
                clip_duration_sec=meta["frames"]/meta["fps"],stop_reason=meta["stop_reason"]))
            (a.output/"视频来源.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
            print(json.dumps(manifest[-1],ensure_ascii=False),flush=True)
    lines = ["# S12 软体圆柱：主要策略手心视频", "",
        f"统一使用预先确定的初态 {seed} 和首个训练种子 {train_seed}，不是从测试结果挑出的最佳例子。固定参考没有训练种子。",
        "每个策略有外观和实际碰撞体两版；本文件夹仅集中视频与目录，不包含策略权重。可整体移动后观看。",
        "完整回合严格通过的短片到拿稳四秒结束；失败片段到首次确定失败的控制帧结束，具体原因见表。是否保持十五秒来自完整回合统计，不由短片长度证明。",
        "", "| 策略 | 显示 | 严格通过 | 物理保持通过 | 完整回合最长严格保持/s | 短片/s | 结束原因 | 视频 |",
        "|---|---|---|---|---:|---:|---|---|"]
    for row in manifest:
        lines.append(f"| {row['label']} | {'外观' if row['mode']=='visual' else '碰撞体'} | "
            f"{'是' if row['strict_success'] else '否'} | {'是' if row['physical_success'] else '否'} | "
            f"{row['best_strict_hold_sec']:.2f} | {row['clip_duration_sec']:.2f} | "
            f"{REASONS[row['stop_reason']]} | [播放]({row['file']}) |")
    lines += ["", "蓝色几何是机械手的实际碰撞形状；橙色是实际可变形物体表面，不是外加的刚体外壳。",
        "视频标注的应变是最大局部单元变形，不是整个物体统一变形百分比。严格失败也可能仍拿着物体，需要区分质量违规和掉落。",
        "`视频来源.json` 保存每段视频、原始轨迹及权重来源哈希。原始轨迹路径指向生成它的工程；移动视频文件夹后仍可播放，但不会自动携带所有原始轨迹。",
        "视频只展示单例；完整成功率、置信区间和失败分解应查阅 S12 正式统计报告。"]
    (a.output/"目录.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status="completed",videos=len(manifest),folder=str(a.output)),ensure_ascii=False))


if __name__=="__main__":
    main()
