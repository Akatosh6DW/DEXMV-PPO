"""Export a few palm-view cylinder policy videos into one portable directory."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_cylinder_s02_source import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for method in ("residual", "guided"):
        run = a.evaluation / method
        rows = json.loads((run / "step_000000000/episodes.json").read_text())
        success = next((i for i, r in enumerate(rows) if r["success30"]), None)
        failure = next((i for i, r in enumerate(rows) if not r["success30"]), None)
        if success is not None:
            cases.append((method, success, rows[success]))
        if failure is not None and (method == "guided" or success is None):
            cases.append((method, failure, rows[failure]))
    videos = []
    for method, index, episode in cases:
        work = a.evaluation / f"render_{method}_{index}"
        subprocess.run([sys.executable, str(ROOT / "scripts/render_s05_clips.py"), "--run", str(a.evaluation / method),
                        "--index", str(index), "--output", str(work), "--task-label", "圆柱_S06_三指夹持",
                        "--header-label", "S06-cylinder"], check=True, cwd=ROOT)
        render = json.loads((work / "render.json").read_text())
        for source in sorted(work.glob("*.mp4")):
            target = a.output / source.name
            shutil.copy2(source, target)
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=width,height,nb_frames", "-of", "json", str(target)], text=True))
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            videos.append(dict(file=target.name, sha256=digest(target), method=method, episode=episode,
                               evaluation_run=str((a.evaluation / method).resolve()), index=index,
                               duration_sec=float(probe["format"]["duration"]), cut_reason=render["reason"],
                               probe=probe, decoded_without_errors=True))
    source = ROOT / "artifacts/orca_benchmark_suite/S06/source_audit/source_contact_sheet.jpg"
    shutil.copy2(source, a.output / "源示范_Adroit处理后重建_非原始人手视频.jpg")
    manifest = dict(evaluation=str(a.evaluation.resolve()), selection_sha256=digest(a.evaluation / "selection.json"),
                    videos=videos, source_image_sha256=digest(source))
    (a.output / "来源与校验.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    lines = ["# 圆柱S06三指抓持视频目录", "", "只保留手心视角，外观和碰撞几何各一版。所有视频来自保存的独立测试状态，不是重新仿真挑选结果。", "",
             "成功案例从起手播放到连续合格夹持4秒，完整40秒测试已验证连续至少30秒；失败案例保留到判定的失败时刻。", "",
             "| 视频 | 测试种子 | 完整评估最长合格保持 | 时长 |", "|---|---:|---:|---:|"]
    for v in videos:
        lines.append(f"| [{v['file']}]({v['file']}) | {v['episode']['seed']} | {v['episode']['pad_hold_sec']:.2f}s | {v['duration_sec']:.2f}s |")
    lines += ["", "案例选取规则：各策略第一个成功案例，另保留BC+PPO第一个失败案例（如果存在）；不能用展示案例代替全部成功率。", "",
              "随附JPG是源样本在Adroit上的重建，不是原始真人视频；其中存在穿模，不能照抄为物理专家。", "",
              "原始统计和详细说明位于主工程 `docs/experiments/S06/`；当前只是小规模可行性验证，不是旧S02完整对比实验的新一轮复现。"]
    (a.output / "目录.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(output=str(a.output), videos=len(videos)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
