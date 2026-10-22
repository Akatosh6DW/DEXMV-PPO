"""Render a fixed illustrative case for every task/method and verify the videos."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for shape in ("cube", "sphere", "cuboid"):
        for arm in ("pure", "bc", "residual"):
            run = args.evaluation/f"{shape}_{arm}_seed{args.training_seed}_terminal"
            meta = json.loads((run/"run_config.json").read_text())
            episodes = json.loads((run/"step_000000000/episodes.json").read_text())
            assert (meta["shape"], meta["arm"], meta["seed"]) == (shape, arm, args.training_seed)
            if not 0 <= args.episode < len(episodes["seeds"]):
                raise ValueError("Episode index is outside the saved evaluation")
            output = args.output/f"{shape}_{arm}"
            command = [sys.executable, "scripts/render_orca_modes.py", "--model", str(run/"model.xml"),
                       "--trace", str(run/"step_000000000/trajectories.npz"),
                       "--episode", str(args.episode), "--output", str(output)]
            print("RENDER", shape, arm, flush=True)
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
            report = json.loads((output/"report.json").read_text())
            streams = {}
            for mode in ("collision", "stl"):
                probe = subprocess.run([
                    "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate:format=duration",
                    "-of", "json", str(output/f"{mode}.mp4")], check=True, capture_output=True, text=True)
                info = json.loads(probe.stdout)
                stream = info["streams"][0]
                assert int(stream["nb_read_frames"]) == report["frames"]
                assert (stream["width"], stream["height"]) == (960, 720)
                numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
                assert numerator/denominator == report["fps"]
                assert abs(float(info["format"]["duration"])-report["seconds"]) < .1
                streams[mode] = info
            records.append(dict(task=shape, arm=arm, training_seed=args.training_seed,
                                reset_seed=episodes["seeds"][args.episode],
                                episode=args.episode, checkpoint=meta["checkpoint"],
                                success=episodes["success30"][args.episode],
                                best_hold_sec=episodes["hold_sec"][args.episode],
                                output=str(output), rendering=report, video_checks=streams))
            manifest = dict(completed=len(records) == 9, cases=records,
                            selection="fixed training seed and reset episode across all methods/tasks; terminal checkpoints, not success-selected clips",
                            caveat="one illustrative rollout is not an estimate of method performance; post-terminal frames are labeled diagnostic-only")
            (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print("VERIFIED", len(records), "cases,", 2*len(records), "videos", flush=True)


if __name__ == "__main__":
    main()
