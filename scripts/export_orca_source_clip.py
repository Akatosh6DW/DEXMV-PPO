"""Preserve the exact source-video frame window associated with the demos."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    screening = ROOT / "artifacts/orca_cube_acquisition/source_screen/report.json"
    data_manifest = ROOT / "artifacts/orca_cube_acquisition/D01_quality_demos/dataset_manifest.json"
    dataset = json.loads(data_manifest.read_text())
    source = next(r for r in json.loads(screening.read_text())["sequences"]
                  if r["sequence"] == dataset["source_sequence"])
    video = Path(source["source_video"])
    assert digest(video) == source["video_sha256"]
    start, end = dataset["source_frames"]
    output = root / "source_video"
    output.mkdir(exist_ok=True)
    target = output / f"hic{source['sequence']}_frames{start}_{end}.mp4"
    manifest_path = output / "manifest.json"
    if target.exists() and not manifest_path.exists():
        raise FileExistsError("Inspect partial source export before retry")
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-an",
               "-vf", f"trim=start_frame={start}:end_frame={end+1},setpts=PTS-STARTPTS",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    if not target.exists():
        subprocess.run(command, check=True)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames,duration", "-of", "json", str(target)], text=True))["streams"][0]
    assert int(probe["nb_read_frames"]) == end-start+1 and probe["r_frame_rate"] == "30/1"
    result = dict(source_video=str(video.relative_to(ROOT)), source_video_sha256=digest(video),
                  source_frame_start=start, source_frame_end_inclusive=end, frames=end-start+1,
                  clip=str(target.relative_to(ROOT)), clip_sha256=digest(target), probe=probe,
                  dataset_manifest_sha256=digest(data_manifest), command=command,
                  note="Original RGB frames, no pose estimator rerun. Control reference has different timing and simulator refinement; this clip is provenance, not an outcome video.")
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["clip_sha256"] == result["clip_sha256"]
    manifest_path.write_text(json.dumps(result, indent=2)+"\n")
    (output / "README.md").write_text(
        "# 示范源视频\n\n"
        f"[HIC20第{start}..{end}帧]({target.name})，共{end-start+1}帧，30fps、2.2秒。\n\n"
        "这是原始RGB片段，未重新执行视频姿态识别。本轮读取对应公开跟踪结果。\n"
        "机器人参考另经重定向、1.5秒时序重采样及仿真校正，不能把参考40秒文件当作40秒源视频。\n"
        "仿真复现53mm整体方块几何，不模拟魔方内部转层；质量、摩擦不是从视频测得。\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
