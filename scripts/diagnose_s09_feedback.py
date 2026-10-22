"""Measure learned feedback effects on saved observations, not rollout success."""

import argparse
import json
from pathlib import Path
import pickle

import jax
import mujoco
import numpy as np

import run_s09_ppo_improvement as improvement
from run_s09_bc_phase_curriculum import KEEP,phase_initialization
from retarget_s09_reference import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--step",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((args.run/"run_config.json").read_text())
    if cfg["method"]!="guided" or cfg["env_config"].get("observation_variant")!="initial_pose_mode_onehot_95_101":
        parser.error("Diagnose the repaired full-target BC+PPO interface only")
    jax.config.update("jax_default_matmul_precision","highest")
    folder=args.run/f"step_{args.step:09d}"
    params=pickle.loads((folder/"params.pkl").read_bytes())
    masked=phase_initialization(params)
    net=improvement.runner.network_factory(dict(state=(123,),privileged_state=(140,)),17)
    factory=improvement.runner.networks.make_inference_fn(net)
    infer=lambda p:jax.jit(lambda obs:factory(p,deterministic=True)({"state":obs},jax.random.PRNGKey(0))[0])
    with np.load(folder/"preview_trajectories.npz",allow_pickle=False) as archive:
        trace=dict(archive)
    actual=np.asarray(infer(params)(trace["obs"]))
    disabled=np.asarray(infer(masked)(trace["obs"]))
    error=float(np.max(np.abs(actual[trace["active"]]-trace["action"][trace["active"]])))
    if error>1e-4:
        raise ValueError(f"Inference does not match saved actions: {error}")
    model=mujoco.MjModel.from_xml_path(str(Path(cfg["input"])/"model.xml"))
    half_range=np.diff(model.actuator_ctrlrange,axis=1).ravel()/2
    first=np.asarray(params[1]["params"]["hidden_0"]["kernel"])
    rows=[]
    for i,seed in enumerate(trace["seeds"]):
        active=trace["active"][i].astype(bool)
        hold=active&(trace["audit"][i,:,10]>0)
        consecutive=hold[1:]&hold[:-1]
        def rms(x):
            return float(np.sqrt(np.mean(x*x))) if x.size else None
        rows.append(dict(seed=int(seed),saved_success30=bool(trace["success30"][i,-1]),
            feedback_effect_rms_rad=rms(((actual[i]-disabled[i])*half_range)[hold]),
            full_temporal_step_rms_rad=rms((np.diff(actual[i],axis=0)*half_range)[consecutive]),
            masked_temporal_step_rms_rad=rms((np.diff(disabled[i],axis=0)*half_range)[consecutive])))
    result=dict(checkpoint_sha256=digest(folder/"params.pkl"),trace_sha256=digest(folder/"preview_trajectories.npz"),
        max_inference_error=error,new_input_weight_rms=float(np.sqrt(np.mean(first[~KEEP]**2))),
        scope="Saved observations only. Masked outputs are NOT a new physical rollout or a measured success rate.",
        rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
