"""Run a uniquely named, fully recorded S12 physical reference probe."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from s12_soft import SoftConfig, SoftGrasp, run_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--young", type=float, default=10000.)
    p.add_argument("--timestep", type=float, default=.001)
    p.add_argument("--damping", type=float, default=.002)
    p.add_argument("--dof", choices=("full", "trilinear", "quadratic"), default="full")
    p.add_argument("--sectors", type=int, default=12)
    p.add_argument("--seconds", type=float, default=24.)
    p.add_argument("--grip-scale", type=float, default=1.)
    p.add_argument("--offset", type=float, nargs=3, default=(0., 0., 0.))
    p.add_argument("--seed", type=int, default=250000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = SoftConfig(young=a.young, timestep=a.timestep, damping=a.damping, dof=a.dof, sectors=a.sectors)
    (a.output/"config.json").write_text(json.dumps(asdict(cfg), indent=2)+"\n")
    start = time.monotonic()
    sim = SoftGrasp(cfg)
    (a.output/"model.xml").write_text(sim.xml)
    print(json.dumps(dict(nq=sim.model.nq, nv=sim.model.nv, vertices=sim.nvert,
                          tetrahedra=len(sim.elements))), flush=True)
    report, trace = run_reference(sim, a.seed, a.seconds, a.grip_scale, offset=a.offset)
    report["wall_sec"] = time.monotonic()-start
    (a.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(a.output/"trajectory.npz", **trace)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
