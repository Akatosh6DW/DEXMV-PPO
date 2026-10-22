"""Package a small, palm-only S05 showcase with source records and decode checks."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.evaluation / "summary.json").read_text())
    assert summary["status"] == "completed"
    args.output.mkdir(parents=True, exist_ok=False)
    records, rows = [], []
    for method, label in (("residual", "视频参考加残差PPO"), ("guided", "BC加PPO"), ("bc_only", "仅BC")):
        run = args.evaluation / method
        episodes = json.loads((run / "step_000000000/episodes.json").read_text())
        successes = [i for i, e in enumerate(episodes) if e["success30"]]
        failures = [i for i, e in enumerate(episodes) if not e["success30"]]
        chosen = [successes[0]] if successes else [0]
        if method == "bc_only" and failures:
            chosen = [failures[0]]
        elif method == "guided" and successes and failures:
            chosen.append(failures[0])
        for index in chosen:
            render = args.evaluation / "renders" / f"{method}_{index}"
            subprocess.run([sys.executable, str(ROOT / "scripts/render_s05_clips.py"),
                            "--run", str(run), "--index", str(index), "--output", str(render),
                            "--policy-label", label], check=True, cwd=ROOT)
            detail = json.loads((render / "render.json").read_text())
            files = []
            for path in sorted(render.glob("*.mp4")):
                target = args.output / path.name
                if target.exists():
                    raise FileExistsError(target)
                shutil.copyfile(path, target)
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"], check=True)
                probe = json.loads(subprocess.check_output([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(target)]))
                records.append(dict(file=target.name, method=method, seed=episodes[index]["seed"],
                                    success30=episodes[index]["success30"], reason=detail["reason"],
                                    duration_sec=float(probe["format"]["duration"]), source=str(render),
                                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(), decoded=True))
                files.append(f"[{'碰撞几何' if '碰撞几何' in target.name else '外观'}]({target.name})")
            rows.append(f"| {label} | {episodes[index]['seed']} | {'成功' if episodes[index]['success30'] else '失败'} | "
                        f"{detail['reason']} | {' / '.join(files)} |")
    lines = ["# S05 方块指腹抓取视频目录", "", "本文件夹内的MP4可整体移动播放，不依赖工程或权重。仅手心角度。", "",
             "- 成功片：从初态到连续合格抓持满4秒；成功标签来自完整40秒评估，不是只检查短片。",
             "- 失败片：截到首个审计失败、终止或已建立抓持失效的时刻，不强行凑够4秒。",
             "- 以20FPS原速播放。外观使用原CAD网格/材质颜色，碰撞版显示实际代理几何。",
             "- 从独立测试中选例，视频数量和选例结果不代表总体成功率。未重新训练纯PPO，不混入旧侧面抓法。", "",
             "| 策略 | 初态种子 | 完整回合 | 截止原因 | 播放 |", "|---|---:|---|---|---|", *rows, "",
             "## 全部独立测试统计", "", "| 方法 | 合格成功数/总数 |", "|---|---:|"]
    labels = {"reference": "仅校正参考回放", "residual": "视频参考+残差PPO", "bc_only": "仅BC", "guided": "BC+PPO"}
    for method, stats in summary["methods"].items():
        lines.append(f"| {labels[method]} | {stats['successes']}/{stats['total']} |")
    lines += ["", "这只是一个训练种子的方块正常条件验证，不代表圆柱、所有扰动或真实硬件成功率。",
              "详细解释：工程中的 `docs/experiments/S05/结果与原因说明.md`。",
              "本目录的 `来源与校验.json` 记录源轨迹、截止原因、文件哈希和解码校验。"]
    (args.output / "目录.md").write_text("\n".join(lines) + "\n")
    (args.output / "来源与校验.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(output=str(args.output), videos=len(records), all_decoded=True)), flush=True)


if __name__ == "__main__":
    main()
