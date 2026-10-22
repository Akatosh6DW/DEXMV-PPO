"""Re-run saved policies using their original evaluation batch and compare previews."""

import argparse
import json
from pathlib import Path
import pickle

import run_s10_job as runner
import jax
import jax.numpy as jp
import numpy as np

from orca_s10_env import OrcaS10Grasp
from s10_mobile_base import OUT
from retarget_s09_reference import digest


def comparison_passes(differences):
    # GPU fusion can round the non-feedback evaluation reward differently by an
    # FP32 ulp. Actions, observations, physics and success must remain bitwise equal.
    return all(error==0 for key,error in differences.items() if key!="reward") and differences.get("reward",0)<=1e-7


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    config=json.loads((a.run/"run_config.json").read_text())
    assert jax.default_backend()==config["backend"]
    jax.config.update("jax_default_matmul_precision","highest")
    jax.config.update("jax_pmap_shmap_merge",False)
    selection=json.loads(Path(config["selection"]).read_text())
    assert digest(Path(config["selection"]))==config["selection_sha256"]
    env=OrcaS10Grasp(OUT/"inputs/model.xml",selection,"residual" if config["method"]=="residual" else "pure")
    net,params=runner.initialize_matched(env,config["seed"],config["std"])
    if config["method"] not in ("reference","collect"):
        params=pickle.loads((a.run/"params.pkl").read_bytes())
        if config.get("checkpoint"):
            assert digest(Path(config["checkpoint"]))==config["checkpoint_sha256"]
    evaluate=runner.evaluator(env,net,reference=config["method"] in ("reference","collect"))
    path=a.run/"step_000000000/preview_trajectories.npz"
    with np.load(path) as z:
        saved=dict(z)
    batches={}; rows=[]
    for index,seed in enumerate(saved["seeds"]):
        batch=config["eval_batch"]
        start=(int(seed)-config["eval_start"])//batch*batch+config["eval_start"]
        if start not in batches:
            batches[start]=jax.device_get(evaluate(params,jp.arange(start,start+batch,dtype=jp.int32)))[1]
        differences={key:float(np.max(np.abs(value[index].astype(float)-batches[start][key][int(seed)-start].astype(float))))
                     for key,value in saved.items() if key!="seeds"}
        rows.append(dict(seed=int(seed),exact=all(v==0 for v in differences.values()),
                         passed=comparison_passes(differences),differences=differences))
    report=dict(status="verified" if all(r["passed"] for r in rows) else "mismatch",run=str(a.run),
                trace_sha256=digest(path),episodes=rows,
                scope="Saved previews: bitwise actions, observations, physics and outcomes; evaluation-only reward absolute tolerance 1e-7; original batch/backend")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    assert report["status"]=="verified"


if __name__=="__main__":
    main()
