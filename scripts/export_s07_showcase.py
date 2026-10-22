"""Export a small, indexed palm-view selection without hiding failed policies."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_cylinder_s02_source import digest
from run_s07_study import OUT, ROOT, write_json

DELIVERY = ROOT / "deliverables/S07_圆柱完整复验_2026-10-12"


def main():
    DELIVERY.mkdir(parents=True, exist_ok=False)
    cases = []
    # Fixed first training seed; no selecting the best seed on held-out results.
    for name in ("ppo_s7101", "guided_s7101", "residual_s7101", "reference"):
        run = OUT / "eval/B0_nominal" / name
        episodes = json.loads((run / "step_000000000/episodes.json").read_text())
        success = next((i for i, r in enumerate(episodes) if r["success30"]), None)
        failure = next((i for i, r in enumerate(episodes) if not r["success30"]), None)
        index = success if success is not None else failure
        cases.append((name, index, episodes[index]))
        if name == "residual_s7101" and success is not None and failure is not None:
            cases.append((name, failure, episodes[failure]))
    videos = []
    for name, index, episode in cases:
        directory = OUT / "renders" / f"{name}_{episode['seed']}"
        subprocess.run([sys.executable, str(ROOT / "scripts/render_s05_clips.py"),
            "--run", str(OUT / "eval/B0_nominal" / name), "--index", str(index), "--output", str(directory),
            "--task-label", f"圆柱_S07_{name}_初态{episode['seed']}", "--header-label", "S07 cylinder"],
            cwd=ROOT, check=True)
        render = json.loads((directory / "render.json").read_text())
        for source in sorted(directory.glob("*.mp4")):
            target = DELIVERY / source.name
            shutil.copy2(source, target)
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=width,height,nb_frames", "-of", "json", str(target)], text=True))
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            videos.append(dict(file=target.name, sha256=digest(target), run=name, episode=episode,
                               duration_sec=float(probe["format"]["duration"]), cut_reason=render["reason"],
                               decoded=True, probe=probe))
    write_json(DELIVERY / "来源与校验.json", dict(selection_sha256=digest(OUT / "selection.json"), videos=videos))
    lines = ["# 圆柱 S07 视频目录", "", "仅手心视角，外观与碰撞几何各一版；不混入方块或旧圆柱数据。", "",
        "取各方法第一个训练种子7101下的第一个成功回合；若没有成功，保留第一个失败回合。残差另保留第一个失败回合（如存在）。这是展示选取规则，不能替代完整成功率。", "",
        "成功片从起手到稳定4秒，完整40秒评估必须另行通过连续30秒；失败片结束于首次确定的失败/丢失已建立合格抓持时刻。", "",
        "| 视频 | 完整评估成功 | 最长严格保持 | 视频时长 |", "|---|---|---:|---:|"]
    for v in videos:
        lines.append(f"| [{v['file']}]({v['file']}) | {'是' if v['episode']['success30'] else '否'} | {v['episode']['pad_hold_sec']:.2f}s | {v['duration_sec']:.2f}s |")
    lines += ["", "完整原始数据、统计和术语解释：主工程 `docs/experiments/S07/`。本目录不含权重，便于整体移动。"]
    (DELIVERY / "目录.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(videos=len(videos), output=str(DELIVERY)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
