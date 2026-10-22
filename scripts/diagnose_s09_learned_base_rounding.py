"""Separate same-state inference error from closed-loop contact amplification."""

import argparse
import json
from pathlib import Path

import run_s09_learned_base_residual as learned
from run_s09_mode_experts import network_factory as expert_factory
from retarget_s09_reference import digest
import jax
import jax.numpy as jp
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--frozen-base",type=Path,required=True)
    p.add_argument("--identity-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    jax.config.update("jax_default_matmul_precision","highest")
    base,params,_ = learned.load_base(args.frozen_base,args.input/"model.xml")
    net = expert_factory(dict(state=(123,),privileged_state=(140,)),17)

    @jax.jit
    def infer(dynamic_params,obs):
        def frame(_,values):
            def action(x):
                logits = net.policy_network.apply(dynamic_params[0],dynamic_params[1],dict(state=x))
                return net.parametric_action_distribution.mode(logits)
            return None,(jax.vmap(action)(values),jax.vmap(base)(values))
        _,(original,frozen) = jax.lax.scan(frame,None,obs.transpose(1,0,2))
        return original.transpose(1,0,2),frozen.transpose(1,0,2)

    records = []
    for name in ("base","zero_residual"):
        path = args.identity_run/(name+"_trace.npz")
        with np.load(path,allow_pickle=False) as z:
            obs,active = z["obs"],z["active"].astype(bool)
        original,frozen = map(np.asarray,infer(params,jp.asarray(obs)))
        error = float(np.max(np.abs(original[active]-frozen[active])))
        np.testing.assert_allclose(original[active],frozen[active],atol=1e-5,rtol=0)
        records.append(dict(source=name,active_frames=int(active.sum()),max_action_error=error,
            trace_sha256=digest(path)))
    result = dict(status="verified_same_state_only",normalized_action_tolerance=1e-5,rows=records,
        base_sha256=digest(args.frozen_base),source_sha256=digest(Path(__file__)),
        implementation_sha256=digest(Path(learned.__file__)),
        limitation="Pointwise inference agreement does not imply bitwise or outcome identity under closed-loop contacts")
    (args.output/"same_state_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
