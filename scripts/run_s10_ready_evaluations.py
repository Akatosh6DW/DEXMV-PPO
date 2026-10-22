"""Evaluate frozen final checkpoints as they become ready; never tune on tests."""

import json
import os
import subprocess
import sys
import time

from s10_mobile_base import OUT, ROOT
from run_s10_study import check


def main():
    protocol=check()
    logs=OUT/"execution"; logs.mkdir(exist_ok=True)
    tasks=[("reference",0),("reference_exact_quality",0)]
    tasks += [(method,seed) for seed in protocol["seeds"] for method in ("ppo","bc","guided","residual")]
    while tasks:
        progress=False
        for method,seed in list(tasks):
            out=OUT/"eval"/f"{method}_s{seed}"
            if out.exists():
                summary=json.loads((out/"summary.json").read_text())
                assert summary["status"]=="completed"
                tasks.remove((method,seed)); progress=True
                continue
            checkpoint=None
            if method not in ("reference","reference_exact_quality"):
                train_method="guided" if method=="bc" else method
                run=OUT/"train"/f"{train_method}_s{seed}"
                step=0 if method=="bc" else protocol["steps"]
                checkpoint=run/f"step_{step:09d}/params.pkl"
                if not checkpoint.exists():
                    continue
                if method!="bc":
                    if not (run/"summary.json").exists():
                        continue
                    summary=json.loads((run/"summary.json").read_text())
                    assert summary["status"]=="completed" and summary["final_step"]==step
            check()
            actual_method="residual" if method=="reference_exact_quality" else "guided" if method=="bc" else method
            command=[sys.executable,str(ROOT/"scripts/run_s10_quality_job.py"),"--output",str(out),
                "--method",actual_method,"--seed",str(seed),"--eval-start",str(protocol["test"][0]),
                "--eval-count",str(protocol["test"][1])]
            if checkpoint:
                command += ["--checkpoint",str(checkpoint)]
            logfile=logs/f"eval_{method}_s{seed}.log"
            with logfile.open("w") as handle:
                process=subprocess.Popen(command,cwd=ROOT,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",XLA_PYTHON_CLIENT_PREALLOCATE="false"),
                    stdout=handle,stderr=subprocess.STDOUT)
                (logs/"evaluation_current.json").write_text(json.dumps(dict(pid=process.pid,command=command,log=str(logfile)),indent=2)+"\n")
                print("START",process.pid,out,flush=True)
                code=process.wait()
            if code:
                raise RuntimeError(f"Evaluation failed {code}: {logfile}")
            summary=json.loads((out/"summary.json").read_text())
            assert summary["status"]=="completed"
            print("DONE",out,json.dumps(summary["success30"]),flush=True)
            tasks.remove((method,seed)); progress=True
        if tasks and not progress:
            # Verify the corresponding training child really exists before waiting.
            current=json.loads((logs/"current.json").read_text())
            if "pid" not in current:
                raise RuntimeError("Training is terminal but required evaluation checkpoints are missing")
            os.kill(current["pid"],0)
            print("WAIT training PID",current["pid"],"pending",tasks,flush=True)
            time.sleep(30)
    (logs/"evaluation_current.json").write_text(json.dumps(dict(status="terminal",phase="eval"))+"\n")


if __name__=="__main__":
    main()
