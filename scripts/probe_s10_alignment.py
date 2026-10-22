"""Development-only initial-pose alignment checks, preserving all failures."""

import argparse
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from s10_mobile_base import CENTER, OUT, MobileCylinder, quaternion, run_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--seconds", type=float, default=12.)
    p.add_argument("--template", type=int, default=0)
    p.add_argument("--rule", choices=("minimal", "gravity"), default="minimal")
    p.add_argument("--roll", type=float, default=0.)
    p.add_argument("--fixed", action="store_true")
    p.add_argument("--prealigned", action="store_true")
    p.add_argument("--seed", type=int, default=210000)
    a = p.parse_args()
    out = OUT / "development" / a.name
    out.mkdir(parents=True, exist_ok=False)
    sim = MobileCylinder(template=a.template, rule=a.rule, roll=a.roll,
                         mobile=not a.fixed, prealigned=a.prealigned)
    from s10_mobile_base import mobile_xml
    (out / "model.xml").write_text(mobile_xml())
    (out / "config.json").write_text(json.dumps(vars(a), indent=2)+"\n")
    rows = []
    started = time.monotonic()
    for seed in range(a.seed, a.seed+a.count):
        rng = np.random.default_rng(seed)
        q = quaternion(Rotation.random(random_state=rng))
        pos = CENTER+rng.uniform(-.002, .002, 3)
        vel = rng.uniform(-.005, .005, 3)*[1, 1, 0]
        row, trace = run_episode(sim, pos, q, vel, seconds=a.seconds, keep_trace=True)
        row.update(seed=seed, object_quaternion=q.tolist(), object_position=pos.tolist())
        rows.append(row)
        np.savez_compressed(out / f"seed_{seed}.npz", **trace)
        (out / "episodes.json").write_text(json.dumps(rows, indent=2)+"\n")
        print(json.dumps(row), flush=True)
    summary = dict(status="completed", elapsed_sec=time.monotonic()-started,
                   success30=float(np.mean([r["success30"] for r in rows])),
                   mean_best_hold_sec=float(np.mean([r["best_hold_sec"] for r in rows])))
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
