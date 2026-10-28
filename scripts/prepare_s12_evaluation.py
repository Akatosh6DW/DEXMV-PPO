"""Bind predetermined final/initial checkpoints after all frozen training jobs finish."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_source(config, job, plan):
    expected = dict(seed=job["seed"],method=job["method"],steps=plan["online_steps"],
        jitter=plan["jitter"],reward_version=plan["reward"],
        eval_start=plan["train_evaluation_seeds"][0],eval_count=len(plan["train_evaluation_seeds"]),
        soft_config=plan["config"],model_sha256=plan["model_sha256"],checkpoint=None,
        bc_phase_only="--bc-phase-only" in job["argv"])
    for field, kind in (("std",float),("learning_rate",float),("evals",int),("physics_workers",int)):
        flag = "--"+field.replace("_","-")
        expected[field] = kind(job["argv"][job["argv"].index(flag)+1])
    if job["method"] == "guided":
        expected["bc_updates"] = int(job["argv"][job["argv"].index("--bc-updates")+1])
        expected["dataset"] = str(Path(job["argv"][job["argv"].index("--dataset")+1]).resolve())
    else:
        expected["dataset"] = None
    differences = {k:dict(expected=v,actual=config.get(k)) for k,v in expected.items()
                   if k not in config or config[k]!=v}
    if differences:
        raise ValueError("Source configuration does not match protocol: "+json.dumps(differences))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.protocol.read_text())
    for name, value in plan["inputs"].items():
        if digest(ROOT/name)!=value:
            raise ValueError("Frozen source mismatch: "+name)
    sources = []
    for job in plan["jobs"]:
        run = Path(job["output"])
        summary = json.loads((run/"summary.json").read_text())
        config = json.loads((run/"run_config.json").read_text())
        if summary["status"]!="completed" or summary["rows"][-1]["step"]!=plan["online_steps"]:
            raise ValueError("Training not finished: "+str(run))
        validate_source(config,job,plan)
        last = run/f"step_{plan['online_steps']:09d}/params.pkl"
        if digest(last)!=digest(run/"params.pkl"):
            raise ValueError("Root checkpoint is not the specified last checkpoint")
        sources.append(dict(label=job["method"], seed=job["seed"], run=run, checkpoint=last))
        if job["method"]=="guided":
            sources.append(dict(label="bc_initial", seed=job["seed"], run=run, checkpoint=run/"bc_params.pkl"))
    sources.append(dict(label="reference", seed=None, run=Path(plan["jobs"][0]["output"]), checkpoint=None))
    jobs = []
    for condition, definition in plan["conditions"].items():
        seeds = definition["seeds"]
        if seeds!=list(range(seeds[0],seeds[0]+len(seeds))):
            raise ValueError("Evaluator expects contiguous, frozen initial seeds")
        for source in sources:
            name = source["label"]+(f"_s{source['seed']}" if source["seed"] is not None else "")
            output = a.protocol.parent/"evaluation"/condition/name
            argv = [sys.executable, str(ROOT/"scripts/evaluate_s12_checkpoint.py"),
                "--run", str(source["run"]), "--condition", condition,
                "--eval-start", str(seeds[0]), "--eval-count", str(len(seeds)),
                "--physics-workers", "4", "--output", str(output)]
            if source["checkpoint"] is None:
                argv.append("--reference")
            else:
                argv += ["--checkpoint", str(source["checkpoint"])]
            jobs.append(dict(name=condition+"_"+name, label=source["label"], train_seed=source["seed"],
                condition=condition, eval_count=len(seeds), output=str(output), checkpoint_sha256=None if source["checkpoint"] is None
                else digest(source["checkpoint"]), checkpoint=None if source["checkpoint"] is None
                else str(source["checkpoint"]), argv=argv))
    output = a.protocol.parent/"evaluation_plan.json"
    with output.open("x") as f:
        json.dump(dict(protocol_sha256=digest(a.protocol), sources_verified=True, jobs=jobs), f, indent=2)
        f.write("\n")
    print(json.dumps(dict(file=str(output), sha256=digest(output), jobs=len(jobs)), indent=2))


if __name__ == "__main__":
    main()
