"""Execute frozen study jobs; bounded concurrency and no silent restarts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--parallel", type=int, default=3, choices=(1,2,3))
    p.add_argument("--evaluation", action="store_true")
    a = p.parse_args()
    plan = json.loads(a.protocol.read_text())
    original = json.loads((a.protocol.parent/"protocol.json").read_text()) if a.evaluation else plan
    if a.evaluation:
        if hashlib.sha256((a.protocol.parent/"protocol.json").read_bytes()).hexdigest()!=plan["protocol_sha256"]:
            raise ValueError("Original protocol changed")
        for job in plan["jobs"]:
            if job["checkpoint"] and hashlib.sha256(Path(job["checkpoint"]).read_bytes()).hexdigest()!=job["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint changed: "+job["name"])
    for name, expected in original["inputs"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError("Frozen input changed: "+name)
    logdir = a.protocol.parent/("evaluation_logs" if a.evaluation else "logs")
    logdir.mkdir(exist_ok=True)
    eventfile = logdir/"execution.jsonl"
    def event(**row):
        row["utc"] = datetime.now(timezone.utc).isoformat()
        with eventfile.open("a") as f:
            f.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
    pending = []
    def completed(summary, job):
        if summary.get("status")!="completed":
            return False
        return (summary["count"]==job["eval_count"] if a.evaluation else
                summary["rows"][-1]["step"]==original["online_steps"])
    for job in plan["jobs"]:
        target = Path(job["output"])
        if target.exists():
            if not (target/"summary.json").exists():
                raise ValueError("Existing unfinished run; inspect manually, never overwrite: "+str(target))
            summary = json.loads((target/"summary.json").read_text())
            if not completed(summary, job):
                raise ValueError("Incomplete budget: "+str(target))
            event(event="already_completed", job=job["name"])
        else:
            pending.append(job)
    live, failed, last_heartbeat = [], [], 0.
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    while pending or live:
        while pending and len(live)<a.parallel and not failed:
            job = pending.pop(0)
            stream = (logdir/(job["name"]+".log")).open("x")
            child = subprocess.Popen(job["argv"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            live.append((job, child, stream))
            event(event="started", job=job["name"], pid=child.pid)
        for item in live[:]:
            job, child, stream = item
            status = child.poll()
            if status is None:
                continue
            stream.close()
            live.remove(item)
            summary_path = Path(job["output"])/"summary.json"
            complete = status==0 and summary_path.exists()
            summary = json.loads(summary_path.read_text()) if complete else None
            complete = complete and completed(summary, job)
            event(event="completed" if complete else "failed", job=job["name"], exit_code=status,
                summary=summary)
            if not complete:
                failed.append(job["name"])
        if failed and not live:
            raise RuntimeError("Training failed; unstarted jobs retained: "+str(failed))
        if live and time.monotonic()-last_heartbeat>60:
            event(event="heartbeat", live=[dict(job=j["name"],pid=c.pid) for j,c,_ in live], pending=len(pending))
            last_heartbeat = time.monotonic()
        if live:
            time.sleep(5)
    event(event="all_evaluation_completed" if a.evaluation else "all_training_completed", count=len(plan["jobs"]))


if __name__ == "__main__":
    main()
