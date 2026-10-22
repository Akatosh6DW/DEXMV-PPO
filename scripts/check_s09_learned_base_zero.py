"""Test zero-residual closed-loop identity against the frozen portfolio."""

import argparse
import json
from pathlib import Path

import run_s09_learned_base_residual as learned
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_mode_experts import network_factory as expert_factory
from retarget_s09_reference import digest
import jax
import jax.numpy as jp
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--frozen-base",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--eval-count",type=int,default=8)
    p.add_argument("--eval-start",type=int,default=130000)
    args = p.parse_args()
    if args.eval_count % 8:
        p.error("Use complete batches of eight")
    args.output.mkdir(parents=True,exist_ok=False)
    jax.config.update("jax_default_matmul_precision","highest")
    action,base_params,_ = learned.load_base(args.frozen_base,args.input/"model.xml")
    env0 = ModeConditionedGrasp(args.input/"model.xml")
    env1 = learned.LearnedBaseGrasp(args.input/"model.xml",base_action=action)
    net0 = expert_factory(dict(state=(123,),privileged_state=(140,)),17)
    net1,params = learned.initialize_residual(learned.residual_factory(action),base_params,14124,.03)
    evaluators = [learned.runner.evaluator(e,n) for e,n in ((env0,net0),(env1,net1))]
    traces,initials,rows = [],[],[]
    seeds = np.arange(args.eval_start,args.eval_start+args.eval_count)
    for label,ev,par in zip(("base","zero_residual"),evaluators,(base_params,params)):
        batches = [jax.device_get(ev(par,jp.asarray(seeds[i:i+8],jp.int32))) for i in range(0,len(seeds),8)]
        initial,trace = jax.tree_util.tree_map(lambda *xs:np.concatenate(xs),*batches)
        row,eps = learned.runner.summarize(trace,seeds)
        traces.append(trace); initials.append(initial); rows.append(row)
        (args.output/(label+"_episodes.json")).write_text(json.dumps(eps,indent=2)+"\n")
        np.savez_compressed(args.output/(label+"_trace.npz"),seeds=seeds,**trace)
    np.testing.assert_array_equal(initials[0],initials[1])
    differences = {}
    for key in ("obs","qpos","qvel","ctrl","mocap","audit","hold","best_hold","success30",
                "success15","initial_penetration","reference_index","desired_ctrl","reward","active","alive"):
        differences[key] = float(np.max(np.abs(traces[0][key].astype(float)-traces[1][key].astype(float))))
    np.testing.assert_array_equal(traces[1]["action"],0)
    np.testing.assert_array_equal(traces[1]["requested_residual"],0)
    result = dict(status="verified" if all(v==0 for v in differences.values()) else "identity_failed",
        differences=differences,rows=rows,base_sha256=digest(args.frozen_base),backend=jax.default_backend(),
        input=str(args.input.resolve()),seed_start=args.eval_start,count=args.eval_count,
        source_sha256=digest(Path(__file__)),implementation_sha256=digest(Path(learned.__file__)))
    (args.output/"identity.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    if result["status"] != "verified":
        raise AssertionError("Zero-residual behavior differs; do not train before diagnosis")


if __name__ == "__main__":
    main()
