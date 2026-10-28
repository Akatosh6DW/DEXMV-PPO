"""Replay identical recorded targets to isolate time-discretization sensitivity."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from s12_soft import SoftConfig, SoftGrasp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timesteps", type=float, nargs="+", default=(.0002,.0001,.00005))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = SoftConfig(**json.loads((a.run / "run_config.json").read_text())["soft_config"])
    reports = []
    with np.load(a.trace) as recorded:
        for dt in a.timesteps:
            sim = SoftGrasp(replace(cfg, timestep=dt))
            sim.reset(offset=recorded["offset"][0])
            rows, error = [], None
            for desired in recorded["desired_ctrl"]:
                try:
                    row = sim.step(desired)
                except FloatingPointError as exc:
                    error = str(exc)
                    break
                row.update(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy(),
                    ctrl=sim.data.ctrl.copy(), vertices=sim.vertices.copy())
                rows.append(row)
                if not row["alive"]:
                    break
            directory = a.output / f"dt_{dt:.8f}"
            directory.mkdir()
            (directory / "model.xml").write_text(sim.xml)
            trace = {key: np.array([r[key] for r in rows]) for key in rows[0]} if rows else {}
            np.savez_compressed(directory / "trajectory.npz", **trace)
            count = len(rows)
            result = dict(config=asdict(sim.cfg), completed_control_steps=count,
                requested_control_steps=len(recorded["time"]), error=error,
                duration_sec=float(sim.data.time),
                min_volume_ratio=float(trace["min_volume_ratio"].min()) if rows else None,
                max_strain=float(trace["max_principal_strain"].max()) if rows else None,
                qpos_max_difference_from_source=float(np.abs(trace["qpos"]-recorded["qpos"][:count]).max()) if rows else None,
                note="Identical target sequence, no policy re-evaluation; not a complete grasp success test")
            (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            reports.append(result)
    (a.output / "report.json").write_text(json.dumps(dict(source_trace=str(a.trace.resolve()),
        source_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(), runs=reports), indent=2) + "\n")


if __name__ == "__main__":
    main()
