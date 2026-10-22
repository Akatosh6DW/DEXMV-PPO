"""Probe an explicit grasp-phase reference without the legacy open-hand blend."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from prepare_hic_orca_object import digest
from probe_hic53_orca import native_rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-frame", type=int, required=True)
    args = p.parse_args()
    reference_path = args.source / "reference.npz"
    with np.load(reference_path) as z:
        fields = {k: z[k] for k in z.files}
    raw, times = fields["raw_ik_ctrl"], fields["source_time"]
    if not 0 <= args.start_frame < len(raw) - 1:
        p.error("Invalid acquisition start")
    phase = times[args.start_frame:] - times[args.start_frame]
    time = np.arange(31) * .05
    ctrl = np.stack([np.interp(time / time[-1], phase / phase[-1], raw[args.start_frame:, k])
                     for k in range(17)], axis=1)
    initial = ctrl[0].copy()
    model = mujoco.MjModel.from_xml_path(str(args.source / "cube.xml"))
    args.output.mkdir(parents=True, exist_ok=False)
    fields.update(ctrl=ctrl, time=time, initial_ctrl=initial,
                  source_frame_start=np.array(args.start_frame))
    np.savez_compressed(args.output / "reference.npz", **fields)
    checks = []
    for seed in range(3):
        trace, check = native_rollout(model, fields["object_position"], fields["object_quaternion"],
                                      ctrl, seed, initial_ctrl=initial)
        checks.append(check)
        np.savez_compressed(args.output / f"native_seed{seed}.npz", **trace)
        print(json.dumps(check), flush=True)
    report = dict(trainable=False, method="source grasp phase without joint-space open reset blend",
                  source=str(args.source.resolve()), source_frames=[args.start_frame, len(raw) - 1],
                  source_duration_sec=float(phase[-1]), target_duration_sec=1.5,
                  initial_ctrl=initial.tolist(), native_checks=checks, command=sys.argv,
                  source_reference_sha256=digest(reference_path),
                  reference_sha256=digest(args.output / "reference.npz"),
                  model_sha256=digest(args.source / "cube.xml"))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ("replay_hic53_reference.py", "probe_hic53_orca.py"):
        (args.output / name).write_bytes(Path(__file__).with_name(name).read_bytes())


if __name__ == "__main__":
    main()
