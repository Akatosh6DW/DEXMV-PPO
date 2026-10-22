"""Fill missing BC/reference showcase cases, without rerunning or altering old experiments."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_cylinder_s02_source import digest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"artifacts/orca_benchmark_suite"
CASES={
    "S05":("S05_补充参考回放_2026-10-12",[("E01_independent/reference","仅参考回放")]),
    "S06":("S06_补充BC与参考_2026-10-12",[("E01_independent/bc_only","仅BC"),("E01_independent/reference","仅参考回放")]),
    "S07":("S07_补充仅BC_2026-10-12",[("eval/B0_nominal/bc_s7101","仅BC")]),
}


def main():
    for study,(name,cases) in CASES.items():
        delivery=ROOT/"deliverables"/name
        if (delivery/"来源与校验.json").exists():
            for row in json.loads((delivery/"来源与校验.json").read_text())["videos"]:
                assert digest(delivery/row["file"])==row["sha256"]
            continue
        delivery.mkdir(exist_ok=True)
        videos=[]
        for relative,label in cases:
            run=DATA/study/relative
            episodes=json.loads((run/"step_000000000/episodes.json").read_text())
            index=next((i for i,e in enumerate(episodes) if e["success30"]),0)
            episode=episodes[index]
            output=DATA/study/"report_supplement_renders"/run.name
            if not (output/"render.json").exists():
                subprocess.run([sys.executable,str(ROOT/"scripts/render_s05_clips.py"),
                    "--run",str(run),"--index",str(index),"--output",str(output),
                    "--task-label",f"{'方块' if study=='S05' else '圆柱'}_{study}_初态{episode['seed']}",
                    "--header-label",study,"--policy-label",label,"--distance",".38"],check=True)
            meta=json.loads((output/"render.json").read_text())
            for source in sorted(output.glob("*.mp4")):
                target=delivery/source.name;shutil.copy2(source,target)
                subprocess.run(["ffmpeg","-v","error","-i",str(target),"-f","null","-"],check=True,
                    stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                duration=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration",
                    "-of","default=nw=1:nk=1",str(target)],text=True))
                videos.append(dict(file=target.name,sha256=digest(target),source_run=str(run.relative_to(ROOT)),
                    episode=episode,duration_sec=duration,cut_reason=meta["reason"],decoded=True))
            shutil.copy2(output/"contact_sheet.jpg",delivery/f"{run.name}_contact_sheet.jpg")
        (delivery/"来源与校验.json").write_text(json.dumps(dict(videos=videos),ensure_ascii=False,indent=2)+"\n")
        lines=[f"# {study}补充视频目录","",
            "补齐该历史实验已有的BC/参考对照视频，不重新训练、不改变结果。取已有固定策略的第一个成功回合，若无成功则取第一个失败；不能用展示代替完整成功率。仅手心视角，外观/碰撞体各一版。",
            "","| 视频 | 完整评估成功 | 最长合格保持 | 剪辑时长 |","|---|---|---:|---:|"]
        for v in videos:
            lines.append(f"| [{v['file']}]({v['file']}) | {v['episode']['success30']} | {v['episode']['pad_hold_sec']:.2f}s | {v['duration_sec']:.2f}s |")
        lines += ["","历史主视频与失败示例仍保留在同实验原视频目录，打包时共同收入该实验的视频分组。"]
        (delivery/"目录.md").write_text("\n".join(lines)+"\n")
        print(study,len(videos),delivery,flush=True)


if __name__=="__main__":
    main()
