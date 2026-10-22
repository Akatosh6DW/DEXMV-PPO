"""Select a single reference/template and alignment rule on development only."""

import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from s10_mobile_base import CENTER, OUT, INPUT, MobileCylinder, quaternion, run_episode, mobile_xml
from retarget_s09_reference import digest


def conditions(start, count):
    for seed in range(start, start+count):
        rng = np.random.default_rng(seed)
        q = quaternion(Rotation.random(random_state=rng))
        pos = CENTER + rng.uniform(-.002, .002, 3)
        vel = rng.uniform(-.005, .005, 3)*[1, 1, 0]
        yield seed, pos, q, vel


def main():
    out = OUT / "development" / "D02_template_search"
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.monotonic()
    for template in range(6):
        for rule in ("minimal", "gravity"):
            sim = MobileCylinder(template=template, rule=rule)
            results = []
            for seed, pos, q, vel in conditions(210000, 8):
                report, _ = run_episode(sim, pos, q, vel, seconds=12.)
                results.append(dict(seed=seed, **report))
            row = dict(template=template, rule=rule, episodes=results,
                       score=float(np.mean([r["best_hold_sec"] for r in results])),
                       peak_force_n=max(r["peak_force_n"] for r in results))
            rows.append(row)
            (out / "short_search.json").write_text(json.dumps(rows, indent=2)+"\n")
            print(json.dumps({k:v for k,v in row.items() if k != "episodes"}), flush=True)
    finalists = sorted(rows, key=lambda r:(r["score"], -r["peak_force_n"]), reverse=True)[:2]
    full = []
    for candidate in finalists:
        sim = MobileCylinder(template=candidate["template"], rule=candidate["rule"])
        results = [dict(seed=seed, **run_episode(sim, pos, q, vel)[0])
                   for seed, pos, q, vel in conditions(210100, 16)]
        full.append(dict(template=candidate["template"], rule=candidate["rule"], episodes=results,
                         success30=float(np.mean([r["success30"] for r in results])),
                         mean_hold=float(np.mean([r["best_hold_sec"] for r in results]))))
        (out / "long_selection.json").write_text(json.dumps(full, indent=2)+"\n")
        print(json.dumps({k:v for k,v in full[-1].items() if k != "episodes"}), flush=True)
    best = max(full, key=lambda r:(r["success30"], r["mean_hold"]))
    inputs = OUT / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    (inputs / "model.xml").write_text(mobile_xml())
    selection = dict(template=best["template"], rule=best["rule"], roll=0.,
                     reference_sha256=digest(INPUT / "reference.npz"), model_sha256=digest(inputs / "model.xml"),
                     short_seeds=list(range(210000, 210008)), long_seeds=list(range(210100, 210116)),
                     selection="single template and rule, development only; no per-test rollout oracle",
                     release_sec=5., move_sec=2., episode_sec=42., hold_sec=30.,
                     workspace_translation=[0., 0., .25])
    (inputs / "selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    (out / "summary.json").write_text(json.dumps(dict(status="completed", selection=selection,
        elapsed_sec=time.monotonic()-started), indent=2)+"\n")
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    main()
