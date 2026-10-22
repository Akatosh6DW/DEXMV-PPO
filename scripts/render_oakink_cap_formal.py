"""Render fixed cap cases and labelled outcomes from saved, audited trajectories."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from oakink_cap_protocol import ROOT, SEEDS
from prepare_hic_orca_object import digest
from render_orca_formal import verify_video


def select_cases(reports):
    cases = []
    for method in ("bc", "pure", "guided", "residual"):
        fixed = dict(run=f"s401_{method}", episode=0, reason="fixed_seed401_reset28500")
        cases.append(fixed)
        outcomes = [(seed, i, success) for seed in SEEDS
                    for i, success in enumerate(reports[f"s{seed}_{method}"]["independent_audit"]["episodes"]["success"])]
        for desired in (False, True):
            selected = next(((seed, i) for seed, i, success in outcomes if bool(success) == desired), None)
            if selected is not None and selected != (401, 0):
                seed, episode = selected
                cases.append(dict(run=f"s{seed}_{method}", episode=episode,
                                  reason="first_" + ("success" if desired else "failure") + "_by_seed_reset_order_not_rate_estimate"))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluations", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    directories, reports = {}, {}
    for seed in SEEDS:
        for method in ("bc", "pure", "guided", "residual"):
            name = f"s{seed}_{method}"
            matches = [root / (name + "__nominal") for root in args.evaluations
                       if (root / (name + "__nominal") / "report.json").exists()]
            if len(matches) != 1:
                raise ValueError("Missing/duplicate cap nominal report")
            directory = matches[0]
            report = json.loads((directory / "report.json").read_text())
            checked = json.loads((directory / "independent_audit.json").read_text())
            assert checked["passed"] and checked["report_sha256"] == digest(directory / "report.json")
            assert not report["development"] and report["seeds"] == list(range(28500, 28564))
            reports[name], directories[name] = report, directory
    cases = select_cases(reports)
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "selection.json").write_text(json.dumps(cases, indent=2) + "\n")
    else:
        assert json.loads((args.output / "selection.json").read_text()) == cases
    results = []
    for case in cases:
        directory, report = directories[case["run"]], reports[case["run"]]
        cfg = json.loads((Path(report["run"]) / "run_config.json").read_text())
        trace, model = directory / "trajectories.npz", Path(cfg["model"]) / "model.xml"
        assert digest(trace) == report["trace_sha256"] and digest(model) == report["model_source_sha256"]
        for azimuth in (225, 315):
            destination = args.output / f"{case['run']}__reset{28500 + case['episode']}__az{azimuth}"
            if not destination.exists():
                command = [sys.executable, "scripts/render_orca_modes.py", "--model", str(model), "--trace", str(trace),
                           "--episode", str(case["episode"]), "--output", str(destination), "--azimuth", str(azimuth),
                           "--elevation", "-18", "--distance", ".42", "--lookat", ".035", ".025", ".245", "--stop-after-termination"]
                subprocess.run(command, cwd=ROOT, env=dict(os.environ, MUJOCO_GL="egl"), check=True, stdout=subprocess.DEVNULL)
            rendering = json.loads((destination / "report.json").read_text())
            assert rendering["trace_sha256"] == report["trace_sha256"] and rendering["episode"] == case["episode"]
            verified = {mode: verify_video(destination / (mode + ".mp4"), rendering) for mode in ("stl", "collision")}
            results.append(dict(**case, azimuth=azimuth, output=str(destination.resolve()), videos=verified,
                                success=report["independent_audit"]["episodes"]["success"][case["episode"]]))
            (args.output / "manifest.json").write_text(json.dumps(dict(cases=results, complete=len(results) == 2 * len(cases),
                 note="Fixed cases plus labelled outcomes; absence of successes is retained, not replaced by mechanism torque tests"), indent=2) + "\n")
            print("CAP_VIDEO", case["run"], case["episode"], azimuth, flush=True)


if __name__ == "__main__":
    main()
