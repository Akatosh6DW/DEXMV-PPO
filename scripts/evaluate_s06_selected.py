"""Freeze development-selected policies before paired, held-out S06 tests."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import mujoco

from prepare_cylinder_s02_source import digest

ROOT = Path(__file__).resolve().parents[1]


def select(run):
    summary = json.loads((run / "summary.json").read_text())
    if summary.get("status") != "completed":
        raise ValueError("Incomplete training run: " + str(run))
    with (run / "metrics.csv").open() as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    eligible = [r for r in rows if r["step"] > 0]
    if not eligible:
        raise ValueError("No actually trained checkpoint: " + str(run))
    best = max(eligible, key=lambda r: (r["success30"], r["pad_hold_sec"], r["step"]))
    checkpoint = run / f"step_{int(best['step']):09d}/params.pkl"
    config = json.loads((run / "run_config.json").read_text())
    return dict(checkpoint=str(checkpoint.resolve()), sha256=digest(checkpoint), development=best,
                bc_updates=config["bc_updates"], std=config["std"], training_budget=config["steps"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--residual-run", type=Path, required=True)
    p.add_argument("--guided-run", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = {name: select(path) for name, path in (("residual", args.residual_run), ("guided", args.guided_run))}
    bc = args.guided_run / "step_000000000/params.pkl"
    bc_updates = selected["guided"]["bc_updates"]
    selected["bc_only"] = dict(checkpoint=str(bc.resolve()), sha256=digest(bc), bc_updates=bc_updates,
        std=selected["guided"]["std"], note=f"{bc_updates} BC updates, no online PPO")
    selection = dict(seeds=list(range(62000, 62000+args.count)), selected=selected,
                     selection_rule="positive-step development success30, then mean hold, then latest step; frozen before test",
                     model_sha256=digest(args.input / "model.xml"), reference_sha256=digest(args.input / "reference.npz"))
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    selection_hash = digest(args.output / "selection.json")
    old_path = ROOT / "artifacts/orca_cylinder_s02/inputs/model.xml"
    old = mujoco.MjModel.from_xml_path(str(old_path))
    new = mujoco.MjModel.from_xml_path(str(args.input / "model.xml"))
    fields = ("body_mass", "body_inertia", "body_pos", "body_quat", "geom_type", "geom_size", "geom_pos", "geom_quat",
              "geom_friction", "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp", "jnt_range",
              "jnt_axis", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
    for field in fields:
        np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
    (args.output / "geometry_invariants.json").write_text(json.dumps(dict(old_model=str(old_path),
        old_model_sha256=digest(old_path), new_model_sha256=selection["model_sha256"], identical_arrays=list(fields),
        changes=dict(impratio=[float(old.opt.impratio), float(new.opt.impratio)],
                     integrator=[int(old.opt.integrator), int(new.opt.integrator)]),
        timestep=float(new.opt.timestep), control_period=.05, equality_constraints=int(new.neq),
        note="Initial object workspace and reference wrist/finger poses changed via reference packet; geometry equality does not mean identical task or physics"), indent=2)+"\n")
    all_rows, summaries, initial = [], {}, None
    for name in ("reference", "residual", "guided", "bc_only"):
        command = [sys.executable, str(ROOT / "scripts/run_s06_job.py"), "--input", str(args.input),
                   "--output", str(args.output / name), "--method", "guided" if name == "bc_only" else name,
                   "--steps", "0", "--eval-count", str(args.count), "--eval-start", "62000",
                   "--seed", "6101", "--height-weight", "5", "--save-all-trajectories"]
        if name in selected:
            if digest(selected[name]["checkpoint"]) != selected[name]["sha256"]:
                raise ValueError("Selected checkpoint changed")
            command += ["--checkpoint", selected[name]["checkpoint"]]
            command += ["--bc-updates", str(selected[name]["bc_updates"]), "--std", str(selected[name]["std"])]
        subprocess.run(command, cwd=ROOT, check=True)
        directory = args.output / name / "step_000000000"
        with np.load(directory / "initial_states.npz") as z:
            if initial is None:
                initial = z["qpos"].copy()
            else:
                np.testing.assert_array_equal(initial, z["qpos"])
        episodes = json.loads((directory / "episodes.json").read_text())
        all_rows += [dict(method=name, **e) for e in episodes]
        k, n = sum(e["success30"] for e in episodes), len(episodes)
        phat, z = k/n, 1.96
        center = (phat+z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
        summaries[name] = dict(successes=k, count=n, success_rate=phat, wilson95=[center-half, center+half],
            mean_hold_sec=float(np.mean([e["pad_hold_sec"] for e in episodes])),
            max_penetration_m=max(e["penetration_m"] for e in episodes),
            peak_force_n=max(e["peak_force_n"] for e in episodes), forbidden_episodes=sum(e["forbidden"] for e in episodes))
    if digest(args.output / "selection.json") != selection_hash:
        raise ValueError("Selection was changed during evaluation")
    with (args.output / "episodes.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    (args.output / "summary.json").write_text(json.dumps(dict(selection_sha256=selection_hash,
        paired_initial_states_equal=True, methods=summaries,
        limitation="One training seed, small independent initial-state sample, not a full comparative benchmark"), indent=2)+"\n")
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
