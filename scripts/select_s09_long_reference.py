"""Select existing bounded corrections using full-horizon DEVELOPMENT replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from retarget_s09_reference import digest
from s09_suspension import SuspendedCylinder, rollout


def score(report):
    return (report["best_hold_sec"] + 5*report["success30"]
            -1000*max(report["max_released_penetration_m"]-.003,0)
            -10*report["forbidden"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--search",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--candidates",type=int,default=16)
    a=p.parse_args()
    if not (a.search/"manifest.json").exists():
        raise ValueError("Short correction search must finish before long selection")
    a.output.mkdir(parents=True,exist_ok=False)
    xml=(a.raw/"model.xml").read_text()
    (a.output/"model.xml").write_text(xml)
    model=SuspendedCylinder(xml).model
    selected=np.r_[0,np.arange(7,17)]
    overall=[]
    for folder in sorted(a.raw.glob("pose_*")):
        target=a.output/folder.name
        target.mkdir()
        with np.load(folder/"reference.npz") as z:
            original=dict(z)
        search=json.loads((a.search/folder.name/"search.json").read_text())
        unique={tuple(r["delta"]):r for r in sorted(search,key=lambda r:r["cost"])}
        ranked=sorted(unique.values(),key=lambda r:r["cost"])
        # Include low-cost candidates and spread across all full-short-hold candidates.
        choices=ranked[:a.candidates//2]
        plateau=[r for r in ranked if r["best_hold_sec"]>=ranked[0]["best_hold_sec"]-.25]
        for i in np.linspace(0,len(plateau)-1,a.candidates,dtype=int):
            if plateau[i] not in choices:
                choices.append(plateau[i])
            if len(choices)>=a.candidates:
                break
        rows=[]
        packets=[]
        for row in choices:
            ctrl=original["ctrl"].copy()
            ctrl[:,selected] += np.minimum(original["time"]/2.,1)[:,None]*np.asarray(row["delta"])
            packet=dict(original,ctrl=np.clip(ctrl,*model.actuator_ctrlrange.T),
                        refinement_delta=np.asarray(row["delta"]),refinement_actuators=selected)
            report,trace=rollout(packet,9041,40.,xml)
            result=dict(candidate=len(rows),search_evaluation=row["evaluation"],delta=row["delta"],
                        nominal=report,score=score(report))
            rows.append(result); packets.append(packet)
        top=sorted(range(len(rows)),key=lambda i:rows[i]["score"],reverse=True)[:4]
        for i in top:
            jittered=[rollout(packets[i],seed,40.,xml,jitter=.001)[0] for seed in (9042,9043,9044)]
            rows[i]["jittered_development"]=jittered
            rows[i]["robust_score"]=min(score(r) for r in [rows[i]["nominal"],*jittered])
        best=max(top,key=lambda i:(rows[i]["robust_score"],rows[i]["score"],-np.linalg.norm(rows[i]["delta"])))
        np.savez_compressed(target/"reference.npz",**packets[best])
        report,trace=rollout(packets[best],9041,40.,xml)
        np.savez_compressed(target/"trace.npz",**trace)
        (target/"selection.json").write_text(json.dumps(dict(best=best,candidates=rows),indent=2)+"\n")
        summary=dict(pose=folder.name,best=best,nominal=report,robust_score=rows[best]["robust_score"],
                     jittered=rows[best]["jittered_development"])
        overall.append(summary)
        (a.output/"report.json").write_text(json.dumps(overall,indent=2)+"\n")
        print(json.dumps(summary),flush=True)
    (a.output/"manifest.json").write_text(json.dumps(dict(status="development_only",raw=str(a.raw),search=str(a.search),
        candidates=a.candidates,search_manifest_sha256=digest(a.search/"manifest.json"),
        selection_seeds=[9041,9042,9043,9044],test_used=False,
        warning="All full-horizon and jittered replays here are development, not a held-out performance estimate"),indent=2)+"\n")


if __name__=="__main__":
    main()
