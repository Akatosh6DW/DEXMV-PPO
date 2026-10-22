"""Full-duration DEVELOPMENT selection, because short holds can hide slipping."""

import json
import time

import numpy as np

from search_s10_alignment import conditions
from s10_mobile_base import OUT, MobileCylinder, run_episode


def main():
    out = OUT / "development" / "D03_long_alignment"
    out.mkdir(parents=True, exist_ok=False)
    choices = [(i, "minimal", 0.) for i in range(6)] + [(0, "gravity", 0.), (1, "gravity", 0.),
                (1, "minimal", -.5), (1, "minimal", .5), (0, "minimal", -.5), (0, "minimal", .5)]
    rows = []
    started = time.monotonic()
    for template, rule, roll in choices:
        sim = MobileCylinder(template=template, rule=rule, roll=roll)
        episodes = [dict(seed=seed, **run_episode(sim, pos, q, vel)[0])
                    for seed, pos, q, vel in conditions(210100, 16)]
        row = dict(template=template, rule=rule, roll=roll, episodes=episodes,
                   success30=float(np.mean([r["success30"] for r in episodes])),
                   mean_hold=float(np.mean([r["best_hold_sec"] for r in episodes])),
                   max_force=max(r["peak_force_n"] for r in episodes))
        rows.append(row)
        (out / "search.json").write_text(json.dumps(rows, indent=2)+"\n")
        print(json.dumps({k:v for k,v in row.items() if k!="episodes"}), flush=True)
    best = max(rows, key=lambda r:(r["success30"], r["mean_hold"], -r["max_force"]))
    selection = json.loads((OUT / "inputs/selection.json").read_text())
    selection.update({k:best[k] for k in ("template", "rule", "roll")})
    selection["selection"] = "D03 full-duration development selection; no held-out outcomes used"
    (OUT / "inputs/selection_v2.json").write_text(json.dumps(selection, indent=2)+"\n")
    (out / "summary.json").write_text(json.dumps(dict(status="completed", selection=selection,
        elapsed_sec=time.monotonic()-started), indent=2)+"\n")


if __name__ == "__main__":
    main()
