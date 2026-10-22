"""Recheck historical statistics and saved traces before placing them in a report archive."""

import csv
import json
from pathlib import Path

import numpy as np

from prepare_cylinder_s02_source import digest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"artifacts/orca_benchmark_suite"


def audit_trace(path,episodes):
    with np.load(path) as z:
        t=dict(z)
    assert all(np.isfinite(v).all() for v in t.values())
    n=len(t["qpos"])
    pair=(t["pad_audit"][...,:2]>0).all(-1)&~(t["pad_audit"][...,2:4]>0).any(-1)
    np.testing.assert_array_equal(pair,t["pad_pair"].astype(bool))
    active=t["active"].astype(bool)
    valid=pair&active&t["alive"].astype(bool)&(t["released"]>0)
    valid&=(t["substep_forbidden"]==0)&(t["substep_penetration"]<=.003)
    valid&=t["cube_z"]>t["spawn_z"]-.01
    count=np.zeros(n,int); hold=[]
    for j in range(800):
        count=np.where(valid[:,j],count+1,0);hold.append(count*.05)
    hold=np.stack(hold,1)
    np.testing.assert_allclose(hold,t["hold"],atol=5e-6)
    for i,e in enumerate(episodes[:n]):
        released=active[i]&(t["released"][i]>0)
        dirty=bool((t["substep_forbidden"][i,released]>0).any())
        depth=float(np.max(t["substep_penetration"][i,released],initial=0))
        success=bool(hold[i].max()>=30 and valid[i,-1] and not dirty and depth<=.003)
        assert success==e["success30"]
        assert abs(hold[i].max()-e["pad_hold_sec"])<5e-6
    return n


def main():
    result=[]
    for study in ("S05","S06","S07"):
        directory=DATA/study
        paths=sorted(directory.glob("eval/*/*/step_000000000/episodes.json")) if study=="S07" else sorted(directory.glob("E01_independent/*/step_000000000/episodes.json"))
        assert len(paths)==(86 if study=="S07" else 4)
        episodes_count,trace_count,initials=0,0,{}
        for path in paths:
            episodes=json.loads(path.read_text())
            run=path.parent.parent
            config=json.loads((run/"run_config.json").read_text())
            for field in ("model","reference"):
                assert digest(Path(config[field]))==config[field+"_sha256"]
            if config.get("checkpoint_sha256"):
                assert digest(Path(config["checkpoint"]))==config["checkpoint_sha256"]
            for name,checksum in config.get("source_sha256",{}).items():
                assert digest(run/"sources"/name)==checksum
            group=run.parent.name if study=="S07" else "normal"
            if group.startswith("A_"):
                group="B0_nominal"
            with np.load(path.parent/"initial_states.npz") as z:
                np.testing.assert_array_equal(initials.setdefault(group,z["qpos"].copy()),z["qpos"])
                np.testing.assert_array_equal(z["seeds"],[e["seed"] for e in episodes])
            trace_count+=audit_trace(path.parent/"preview_trajectories.npz",episodes)
            episodes_count+=len(episodes)
        if study=="S07":
            assert episodes_count==3424 and trace_count==1604
            with (directory/"report/episodes.csv").open() as f:
                flat=list(csv.DictReader(f))
            assert len(flat)==episodes_count
            for path in paths:
                run=path.parent.parent
                source=json.loads(path.read_text())
                saved=[x for x in flat if x["benchmark"]==run.parent.name and x["run"]==run.name]
                assert len(saved)==len(source)
                assert [x["success30"]=="True" for x in saved]==[x["success30"] for x in source]
        else:
            summary=json.loads((directory/"E01_independent/summary.json").read_text())
            for path in paths:
                rows=json.loads(path.read_text())
                assert sum(x["success30"] for x in rows)==summary["methods"][path.parent.parent.name]["successes"]
        row=dict(study=study,episode_metrics=episodes_count,recomputed_saved_traces=trace_count,
            run_count=len(paths),paired_initials=True,raw_counts_match_report=True,input_and_checkpoint_hashes=True)
        result.append(row);print(json.dumps(row),flush=True)
    target=DATA/"S08/report"
    target.mkdir(exist_ok=True)
    (target/"historical_recheck.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
