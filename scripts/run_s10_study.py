"""Frozen S10 study: identical movement, tasks and online budgets across methods."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from retarget_s09_reference import digest
from s10_mobile_base import OUT, INPUT, ROOT

SOURCES = ("s10_mobile_base.py", "orca_s10_env.py", "run_s10_job.py", "run_s10_study.py",
           "s09_suspension.py", "orca_native_physics.py", "run_s09_job.py", "train_orca_matched.py",
           "train_orca_comparison.py")


def freeze():
    path = OUT/"protocol.json"
    if path.exists():
        raise FileExistsError(path)
    selection = OUT/"inputs/selection_v2.json"
    assert selection.exists(), "Finish long-duration development selection first"
    files = [selection, OUT/"inputs/model.xml", INPUT/"reference.npz", *[ROOT/"scripts"/s for s in SOURCES]]
    protocol = dict(version="S10-R0", seeds=[22101,22102,22103], steps=163840,
        methods=["ppo","guided","residual","reference"], diagnostic="bc",
        collection=[212000,64], development=[211000,16], test=[220000,128],
        recipes=dict(ppo=dict(lr=3e-4,std=.3), guided=dict(lr=3e-6,std=.05,bc_updates=20000),
                     residual=dict(lr=3e-5,std=.1)),
        final_selection="Last checkpoint only; development checkpoints retained; test never selects weights",
        controller="Same frozen template and continuous 6DoF approach for every arm",
        physics="Dynamic free root plus compliant weld to noncolliding mocap target; cylinder never welded",
        criterion="42s, release5s, 30s strict continuous hold and valid at end; unchanged S09 contact gates",
        limits="Ideal Cartesian positioner, not actual robot-arm dynamics or proof of reachability/safety",
        hashes={str(f.relative_to(ROOT)):digest(f) for f in files})
    path.write_text(json.dumps(protocol,indent=2)+"\n")
    print(json.dumps(protocol),flush=True)


def check():
    protocol=json.loads((OUT/"protocol.json").read_text())
    for file,sha in protocol["hashes"].items():
        assert digest(ROOT/file)==sha,"Frozen input changed: "+file
    return protocol


def jobs(phase, p):
    base=[sys.executable,str(ROOT/"scripts/run_s10_job.py")]
    if phase=="collect":
        out=OUT/"demonstrations"
        yield out,base+["--output",str(out),"--method","collect","--eval-start",str(p["collection"][0]),
                       "--eval-count",str(p["collection"][1])],0
    elif phase=="train":
        for seed in p["seeds"]:
            for method in ("ppo","guided","residual"):
                out=OUT/"train"/f"{method}_s{seed}"
                recipe=p["recipes"][method]
                command=base+["--output",str(out),"--method",method,"--seed",str(seed),
                    "--steps",str(p["steps"]),"--eval-start",str(p["development"][0]),
                    "--eval-count",str(p["development"][1]),"--learning-rate",str(recipe["lr"]),
                    "--std",str(recipe["std"])]
                if method=="guided":
                    command += ["--dataset",str(OUT/"demonstrations/trajectories.npz"),
                                "--bc-updates",str(recipe["bc_updates"])]
                yield out,command,p["steps"]
    elif phase=="eval":
        for method in ("reference","ppo","guided","residual"):
            for seed in ([0] if method=="reference" else p["seeds"]):
                out=OUT/"eval"/f"{method}_s{seed}"
                command=base+["--output",str(out),"--method",method,"--eval-start",str(p["test"][0]),
                              "--eval-count",str(p["test"][1]),"--seed",str(seed)]
                if method!="reference":
                    run=OUT/"train"/f"{method}_s{seed}"
                    summary=json.loads((run/"summary.json").read_text())
                    assert summary["status"]=="completed" and summary["final_step"]==p["steps"]
                    command += ["--checkpoint",str(run/f"step_{p['steps']:09d}/params.pkl")]
                yield out,command,0
        for seed in p["seeds"]:
            out=OUT/"eval"/f"bc_s{seed}"
            # Guided step zero is the independently saved BC actor, before online updates.
            checkpoint=OUT/"train"/f"guided_s{seed}"/"step_000000000/params.pkl"
            yield out,base+["--output",str(out),"--method","guided","--seed",str(seed),
                "--eval-start",str(p["test"][0]),"--eval-count",str(p["test"][1]),
                "--checkpoint",str(checkpoint)],0
    else:
        raise ValueError(phase)


def run(phase, limit):
    p=check()
    logs=OUT/"execution"; logs.mkdir(exist_ok=True)
    count=0
    for out,command,expected in jobs(phase,p):
        if out.exists():
            summary=json.loads((out/"summary.json").read_text())
            assert summary["status"]=="completed" and summary["final_step"]==expected
            continue
        check()
        log=logs/("_".join(out.relative_to(OUT).parts)+".log")
        env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",XLA_PYTHON_CLIENT_PREALLOCATE="false")
        with log.open("w") as handle:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            (logs/"current.json").write_text(json.dumps(dict(pid=process.pid,command=command,log=str(log)),indent=2)+"\n")
            print("START",process.pid,out,flush=True)
            code=process.wait()
        if code:
            raise RuntimeError(f"Job failed ({code}): {log}")
        summary=json.loads((out/"summary.json").read_text())
        assert summary["status"]=="completed" and summary["final_step"]==expected
        with (logs/"iterations.jsonl").open("a") as f:
            f.write(json.dumps(dict(run=str(out),command=command,success30=summary["success30"],
                hold_sec=summary["hold_sec"], final_quality=summary["rows"][-1],
                next_action="Execute next frozen matched recipe; no reward/test-driven tuning"))+"\n")
        print("DONE",out,json.dumps(summary["success30"]),flush=True)
        count+=1
        if limit and count>=limit:
            break
    (logs/"current.json").write_text(json.dumps(dict(status="terminal",phase=phase,new_jobs=count))+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("freeze","collect","train","eval"))
    p.add_argument("--limit",type=int,default=0)
    a=p.parse_args()
    freeze() if a.phase=="freeze" else run(a.phase,a.limit)
