"""Compile and execute reconstructed geometry with the unchanged residual interface."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np

from orca_guidance_recheck import REFERENCE, make_evaluator, score
from orca_residual_env import OrcaResidualGrasp
from train_orca_residual import initialize_zero_mean


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    source=args.candidate/'cube.xml'
    proof=json.loads((args.validation/'report.json').read_text())
    assert proof['geometry_and_native_pass']
    assert proof['model_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_default_matmul_precision','highest')
    jax.config.update('jax_compilation_cache_dir',str(Path('artifacts/orca30/jax_cache').resolve()))
    env=OrcaResidualGrasp(source,'residual',REFERENCE,.08)
    net,params=initialize_zero_mean(100,.7)
    start=time.monotonic()
    trace=jax.device_get(make_evaluator(env,net,[9100,9101])(params))
    assert all(np.isfinite(v).all() for v in trace.values())
    row,holds,success=score(trace,env)
    assert np.max(np.abs(trace['requested_residual']))==0
    assert np.max(np.abs(np.diff(trace['ctrl'],axis=1)))<=.10001
    result=dict(pass_mjx=True,backend=jax.default_backend(),model_sha256=proof['model_sha256'],
                validation_report=str((args.validation/'report.json').resolve()),
                elapsed_sec=time.monotonic()-start,seeds=[9100,9101],**row,
                note='Zero-residual physical smoke; no training or performance selection')
    np.savez_compressed(args.output/'trajectories.npz',**trace)
    (args.output/'episodes.json').write_text(json.dumps(dict(seeds=[9100,9101],hold_sec=holds.tolist(),success30=success.tolist())))
    (args.output/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
