"""Isolate friction-constraint drift from policy quality; never change old scenes."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from refine_s05_reference import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter", choices=("noslip_iterations", "impratio"), default="noslip_iterations")
    parser.add_argument("--values", nargs="+", type=float,
                        help="Explicit narrow solver sweep; defaults to the original broad ablation")
    args = parser.parse_args()
    values = args.values or ((0, 1, 5, 10) if args.parameter == "noslip_iterations" else (1, 2, 5, 10))
    if any(not np.isfinite(v) or v < 0 for v in values):
        parser.error("Solver values must be finite and nonnegative")
    if args.parameter == "noslip_iterations":
        if any(v != int(v) for v in values):
            parser.error("NoSlip iteration counts must be integers")
        values = [int(v) for v in values]
    elif any(v <= 0 for v in values):
        parser.error("impratio must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.input / "reference.npz", allow_pickle=False) as archive:
        packet = {k: archive[k].copy() for k in archive.files}
    reports = []
    for iterations in values:
        prefix = "noslip" if args.parameter == "noslip_iterations" else "impratio"
        directory = args.output / f"{prefix}_{iterations:g}"
        directory.mkdir()
        root = ET.parse(args.input / "model.xml").getroot()
        root.find("option").set(args.parameter, str(iterations))
        ET.ElementTree(root).write(directory / "model.xml", encoding="unicode")
        (directory / "reference.npz").write_bytes((args.input / "reference.npz").read_bytes())
        model = mujoco.MjModel.from_xml_path(str(directory / "model.xml"))
        samples = []
        for seed in range(5030, 5034):
            report, trace = evaluate(model, packet, packet["ctrl"], seed, 40.)
            samples.append(report)
            np.savez_compressed(directory / f"probe_{seed}.npz", **trace)
        # Negative control: explicitly open the hand after 5 seconds. NoSlip must
        # not behave like an invisible weld or keep an unsupported object fixed.
        opening = np.tile(packet["ctrl"][-1], (800, 1))
        opening[:len(packet["ctrl"])] = packet["ctrl"]
        opening[100:] = packet["initial_ctrl"]
        negative, _ = evaluate(model, packet, opening, 5030, 40.)
        # A separate negative-control model removes hand/object contacts only;
        # floor and support remain. No positive rollout uses this intervention.
        detached = mujoco.MjModel.from_xml_path(str(directory / "model.xml"))
        cube = detached.geom("cube").id
        detached.geom_contype[cube], detached.geom_conaffinity[cube] = 0, 4
        no_contact, _ = evaluate(detached, packet, packet["ctrl"], 5030, 8.)
        row = dict(solver_parameter=args.parameter, solver_value=iterations,
                   episodes=samples, open_hand_control=negative, no_hand_contact_control=no_contact,
                   mean_pad_hold_sec=float(np.mean([s["pad_hold_sec"] for s in samples])),
                   peak_force_n=max(s["max_force_n"] for s in samples),
                   max_penetration_m=max(s["max_penetration_m"] for s in samples),
                   opening_drops=negative["duration_sec"] < 40,
                   no_contact_drops=no_contact["duration_sec"] < 8)
        (directory / "report.json").write_text(json.dumps(row, indent=2) + "\n")
        reports.append(row)
        print(json.dumps(row), flush=True)
    (args.output / "report.json").write_text(json.dumps(dict(
        input=str(args.input), mujoco_version=mujoco.__version__, cases=reports,
        scope="Numerical solver ablation; not a PPO improvement or original-benchmark result",
        docs="https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip"), indent=2) + "\n")


if __name__ == "__main__":
    main()
