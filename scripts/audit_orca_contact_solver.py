"""Separate solver convergence from soft-contact creep on identical controls."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from orca_residual_controls import load_reference
from prepare_hic_orca_object import digest
from probe_hic53_orca import native_rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--diagnostic-commands", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fine-step", action="store_true")
    p.add_argument("--iteration-sweep", action="store_true")
    p.add_argument("--solver-sweep", action="store_true")
    args = p.parse_args()
    if sum((args.fine_step, args.iteration_sweep, args.solver_sweep)) > 1:
        p.error("Select a single controlled sweep")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    with np.load(args.reference) as z:
        packet = {k: z[k] for k in z.files}
    model = mujoco.MjModel.from_xml_path(str(args.model))
    base, _ = load_reference(args.reference, *model.actuator_ctrlrange.T, packet["initial_ctrl"], .05, 800, .1)
    with np.load(args.diagnostic_commands) as z:
        correction = z["ctrl"].copy()
    variants = ([("elliptic_32_1_dt5", 32, 1, 1., .005),
                 ("elliptic_32_1_dt2p5", 32, 1, 1., .0025),
                 ("elliptic_32_5_dt2p5", 32, 1, 5., .0025),
                 ("elliptic_32_10_dt2p5", 32, 1, 10., .0025)] if args.fine_step else
                [("pyramid_8_1", 8, 0, 1., .005), ("pyramid_32_1", 32, 0, 1., .005),
                 ("elliptic_32_1", 32, 1, 1., .005), ("elliptic_32_5", 32, 1, 5., .005),
                 ("elliptic_32_10", 32, 1, 10., .005)])
    if args.iteration_sweep:
        variants = [(f"elliptic_{n}_1_dt5", n, 1, 1., .005) for n in (32, 16, 8)]
    if args.solver_sweep:
        variants = [("newton32", 32, 1, 1., .005)] + [(f"cg{n}", n, 1, 1., .005) for n in (32, 64, 128, 256)]
    results, started = [], time.monotonic()
    for name, iterations, cone, impratio, timestep in variants:
        model.opt.iterations, model.opt.cone, model.opt.impratio = iterations, cone, impratio
        model.opt.timestep = timestep
        model.opt.solver = mujoco.mjtSolver.mjSOL_CG if name.startswith("cg") else mujoco.mjtSolver.mjSOL_NEWTON
        for label, commands in (("video_only", base), ("bounded_search_control", correction)):
            variant_started = time.monotonic()
            checks = []
            for seed in (1300, 1301, 1302):
                trace, check = native_rollout(model, packet["object_position"], packet["object_quaternion"],
                                              commands, seed, initial_ctrl=packet["initial_ctrl"])
                checks.append(check)
                if seed == 1300:
                    np.savez_compressed(args.output / f"{name}_{label}.npz", **trace)
            row = dict(name=name, iterations=iterations, cone=cone, impratio=impratio, timestep=timestep,
                       solver=int(model.opt.solver), elapsed_sec=time.monotonic() - variant_started,
                       control=label, checks=checks,
                       mean_hold_sec=float(np.mean([c["best_hold_sec"] for c in checks])),
                       success30=float(np.mean([c["success30"] for c in checks])))
            results.append(row)
            print(json.dumps(row), flush=True)
    report = dict(kind="native solver sensitivity, not trained PPO", results=results,
                  elapsed_sec=time.monotonic() - started, model_sha256=digest(args.model),
                  reference_sha256=digest(args.reference), commands_sha256=digest(args.diagnostic_commands),
                  no_mesh_mass_friction_or_control_changes=True,
                  note="impratio modifies soft-constraint impedance, not measured material friction; requires GPU and sensitivity validation")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
