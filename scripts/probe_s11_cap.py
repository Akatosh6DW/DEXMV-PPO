"""Small paired mechanics probe before spending an RL training budget."""

import argparse
import hashlib
import json

import numpy as np

from s11_cap import CapPilot, OUT, SOURCE, REFERENCE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name",required=True)
    p.add_argument("--count",type=int,default=2)
    p.add_argument("--grip-times",type=float,nargs="+",default=[0.,1.,2.])
    p.add_argument("--twist-seconds",type=float,default=14.)
    a=p.parse_args()
    out=OUT/a.name
    out.mkdir(parents=True,exist_ok=False)
    summary=[]
    for moving in (False,True):
        for grip in a.grip_times:
            sim=CapPilot(moving=moving,grip_time=grip,twist_seconds=a.twist_seconds)
            group=out/f"{'moving' if moving else 'fixed'}_grip{grip:g}"
            group.mkdir()
            (group/"model.xml").write_text(sim.xml)
            for seed in range(240000,240000+a.count):
                sim.reset(seed)
                rows=[]
                for _ in range(round(sim.duration/sim.dt)):
                    row=sim.step();rows.append(row)
                    if row["floor_contact"] or row["peak_force_n"]>100:
                        break
                arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
                np.savez_compressed(group/f"seed{seed}.npz",**arrays)
                record=dict(moving=moving,grip_time=grip,seed=seed,success=rows[-1]["success"],
                    released=rows[-1]["released"],max_degrees=float(np.rad2deg(arrays["angle"].max())),
                    final_degrees=float(np.rad2deg(rows[-1]["angle"])),final_lift_m=rows[-1]["lift"],
                    best_hold_sec=sim.best_hold*sim.dt,final_hold_sec=rows[-1]["hold_sec"],
                    initial_depth_m=sim.initial_depth,max_depth_m=sim.max_depth,peak_force_n=sim.max_force,
                    ever_bad=bool(sim.ever_bad),duration_sec=rows[-1]["time"],trace=str((group/f"seed{seed}.npz").relative_to(out)))
                summary.append(record)
                (out/"episodes.json").write_text(json.dumps(summary,indent=2)+"\n")
                print(json.dumps(record),flush=True)
    report=dict(status="completed",kind="pregrasp mechanics feasibility, not a learned policy comparison",
        source_model_sha256=hashlib.sha256((SOURCE/"model.xml").read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        config=vars(a),episodes=summary)
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
