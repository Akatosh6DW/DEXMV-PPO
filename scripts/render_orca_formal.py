"""Render audited fixed cases and explicitly labelled failures, palm-facing."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from orca_formal_protocol import SEEDS
from prepare_hic_orca_object import digest

ROOT = Path(__file__).resolve().parents[1]


def select_cases(reports):
    cases = []
    for method in ("bc", "ppo", "guided", "residual"):
        for tactile in ("off", "on"):
            ordered = [reports[f"s{seed}_{method}_{tactile}"] for seed in SEEDS]
            first = ordered[0]
            cases.append(dict(run=f"s301_{method}_{tactile}", episode=0, reason="fixed_seed301_reset25000"))
            failed = [(seed, i) for seed, report in zip(SEEDS, ordered)
                      for i, success in enumerate(report["episodes"]["success30"]) if not success]
            if failed and failed[0] != (301, 0):
                seed, episode = failed[0]
                cases.append(dict(run=f"s{seed}_{method}_{tactile}", episode=episode,
                                  reason="first_failure_by_seed_and_reset_order"))
            if method == "residual" and not first["episodes"]["success30"][0]:
                successes = [(seed, i) for seed, report in zip(SEEDS, ordered)
                             for i, success in enumerate(report["episodes"]["success30"]) if success]
                if successes:
                    seed, episode = successes[0]
                    cases.append(dict(run=f"s{seed}_{method}_{tactile}", episode=episode,
                                      reason="explicitly_success_selected_residual_example_not_performance_estimate"))
    return cases


def verify_video(path, rendering):
    result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                             "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate:format=duration",
                             "-of", "json", str(path)], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    assert int(stream["nb_read_frames"]) == rendering["frames"]
    assert (stream["width"], stream["height"]) == (960, 720)
    num, den = map(int, stream["r_frame_rate"].split("/"))
    assert num / den == rendering["fps"] == 20
    assert abs(float(info["format"]["duration"]) - rendering["seconds"]) < .1
    return dict(sha256=digest(path), ffprobe=info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    reports, directories = {}, {}
    for seed in SEEDS:
        for method in ("bc", "ppo", "guided", "residual"):
            for tactile in ("off", "on"):
                name = f"s{seed}_{method}_{tactile}"
                matches = [root / (name + "__nominal") for root in args.evaluations
                           if (root / (name + "__nominal") / "report.json").exists()]
                if len(matches) != 1:
                    raise ValueError("Missing or duplicate nominal evaluation: " + name)
                directory = matches[0]
                report = json.loads((directory / "report.json").read_text())
                audit = json.loads((directory / "independent_audit.json").read_text())
                assert audit["passed"] and audit["report_sha256"] == digest(directory / "report.json")
                assert not report["development_smoke"] and report["seeds"] == list(range(25000, 25064))
                reports[name] = report
                directories[name] = directory
    cases = select_cases(reports)
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "selection.json").write_text(json.dumps(cases, indent=2) + "\n")
    else:
        assert json.loads((args.output / "selection.json").read_text()) == cases
    results = []
    for case in cases:
        name, episode = case["run"], case["episode"]
        directory = directories[name]
        destination = args.output / f"{name}__reset{25000 + episode}"
        trace = directory / "trajectories.npz"
        report = reports[name]
        assert digest(trace) == report["trace_sha256"]
        if not destination.exists():
            command = [sys.executable, "scripts/render_orca_modes.py", "--model", str(directory / "model.xml"),
                       "--trace", str(trace), "--episode", str(episode), "--output", str(destination),
                       "--azimuth", "225", "--elevation", "-18", "--lookat", ".035", ".025", ".245",
                       "--distance", ".38", "--stop-after-termination"]
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, MUJOCO_GL="egl"), check=True,
                           stdout=subprocess.DEVNULL)
        rendering = json.loads((destination / "report.json").read_text())
        assert rendering["trace_sha256"] == report["trace_sha256"] and rendering["episode"] == episode
        checks = {mode: verify_video(destination / f"{mode}.mp4", rendering) for mode in ("stl", "collision")}
        results.append(dict(**case, reset_seed=25000 + episode,
                            success30=report["episodes"]["success30"][episode],
                            hold_sec=report["episodes"]["hold_sec"][episode],
                            output=str(destination.resolve()), videos=checks, rendering=rendering))
        (args.output / "manifest.json").write_text(json.dumps(dict(cases=results, complete=len(results) == len(cases),
                                                                  note="Fixed cases plus labelled failures; videos are illustrations, not estimates of success rate"), indent=2) + "\n")
        print("VERIFIED_VIDEO", name, episode, case["reason"], flush=True)


if __name__ == "__main__":
    main()
