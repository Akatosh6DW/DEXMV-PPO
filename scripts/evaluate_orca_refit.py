"""Paired geometry/checkpoint evaluation, reusing one compiled graph per arm/model."""

import argparse
import csv
import json
from pathlib import Path
import pickle
import time

import jax
import numpy as np

from orca_guidance_recheck import REFERENCE, score
from orca_refit_env import OrcaRefitGrasp
from train_orca_refit import ROOT, check_gates, digest, evaluator
from train_orca_residual import initialize_zero_mean


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--eval-start',type=int,default=9200)
    p.add_argument('--eval-count',type=int,default=16)
    args=p.parse_args()
    jobs=json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'manifest.json').write_text(json.dumps(jobs,indent=2))
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    jax.config.update('jax_default_matmul_precision','highest')
    jax.config.update('jax_compilation_cache_dir',str(ROOT/'artifacts/orca30/jax_cache'))
    assert jax.default_backend()=='gpu'
    seeds=list(range(args.eval_start,args.eval_start+args.eval_count))
    rows=[];cache={}
    for job in jobs:
        started=time.monotonic()
        model=Path(job['model']).resolve()
        if job.get('gate'):
            check_gates(model,Path(job['gate']))
        else:
            assert digest(model)=='603c7f14b0cdc4f190699757abfea36cfe0f677a08219619963431160dd05e45'
        arm=job['arm']
        net,params=initialize_zero_mean(job.get('seed',101),.7)
        checkpoint=Path(job['checkpoint']) if job.get('checkpoint') else None
        if checkpoint:
            params=pickle.loads(checkpoint.read_bytes())
        # Match input leaf types for prior and checkpoints to avoid weak-type recompiles.
        params=jax.tree_util.tree_map(np.asarray,jax.device_get(params))
        key=(str(model),'pure' if arm=='bc' else arm)
        if key not in cache:
            env=OrcaRefitGrasp(model,arm,REFERENCE)
            cache[key]=(env,evaluator(env,net,seeds))
        env,evaluate=cache[key]
        trace=jax.device_get(evaluate(params))
        assert all(np.isfinite(v).all() for v in trace.values())
        row,holds,success=score(trace,env)
        row.update(name=job['name'],arm=arm,elapsed_sec=time.monotonic()-started)
        rows.append(row)
        folder=args.output/job['name'];folder.mkdir()
        np.savez_compressed(folder/'trajectories.npz',**trace)
        (folder/'episodes.json').write_text(json.dumps(dict(seeds=seeds,hold_sec=holds.tolist(),success30=success.tolist())))
        (folder/'config.json').write_text(json.dumps(dict(**job,model_sha256=digest(model),
            checkpoint_sha256=digest(checkpoint) if checkpoint else None,reference_sha256=digest(REFERENCE),
            canonical_numpy_parameters=True),indent=2))
        with (args.output/'metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
