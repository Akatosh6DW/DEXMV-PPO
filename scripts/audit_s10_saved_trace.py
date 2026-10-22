"""Replay saved controls in independent native physics and check stored outcomes."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Aero-Hand/sim_rl/mujoco_playground"))

import jax
import mujoco
import numpy as np

from s10_mobile_base import CENTER, OUT, MobileCylinder
from retarget_s09_reference import digest
from orca_s10_env import OrcaS10Grasp
from orca_native_physics import STATE_SPEC


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    config=json.loads((a.trace.parent.parent/"run_config.json").read_text())
    if jax.default_backend()!=config["backend"]:
        raise ValueError("Replay must reproduce the recorded backend's RNG and reset")
    jax.config.update("jax_default_matmul_precision","highest")
    jax.config.update("jax_pmap_shmap_merge",False)
    selection=json.loads((OUT/"inputs/selection_v2.json").read_text())
    env=OrcaS10Grasp(OUT/"inputs/model.xml",selection,
                    "residual" if config["method"]=="residual" else "pure")
    reset=jax.jit(jax.vmap(env.reset))
    sim=MobileCylinder((OUT/"inputs/model.xml").read_text(),template=selection["template"],
                       rule=selection["rule"],roll=selection["roll"])
    results=[]
    with np.load(a.trace) as z:
        for i,seed in enumerate(z["seeds"]):
            batch=config["eval_batch"]
            offset=(int(seed)-config["eval_start"])//batch*batch+config["eval_start"]
            keys=jax.vmap(jax.random.PRNGKey)(jax.numpy.arange(offset,offset+batch))
            states=jax.device_get(reset(keys))
            state=jax.tree_util.tree_map(lambda v:v[int(seed)-offset],states)
            sim.reset(state.info["cube_spawn_pos"],state.data.initial_object_quat)
            mujoco.mj_setState(sim.model,sim.data,
                np.asarray(state.data.integration_words).copy().view(np.float64),STATE_SPEC)
            sim.initial_penetration=float(state.info["initial_penetration"])
            position_error=0.; audit_error=0.; rows=[]
            for t in np.flatnonzero(z["active"][i]):
                row=sim.step(z["desired_ctrl"][i,t])
                rows.append(row)
                position_error=max(position_error,float(np.max(np.abs(sim.data.qpos.astype(np.float32)-z["qpos"][i,t]))))
                expected=np.r_[row["loads"],row["forbidden"],row["penetration_m"],row["peak_force_n"],
                    row["rope_force_n"],row["alive"],row["released"],row["valid"],row["time"]].astype(np.float32)
                audit_error=max(audit_error,float(np.max(np.abs(expected-z["audit"][i,t]))))
            same=bool(rows[-1]["success30"]==bool(z["success30"][i,-1]))
            results.append(dict(seed=int(seed),steps=len(rows),qpos_max_error=position_error,
                                audit_max_error=audit_error,success_matches=same,
                                success30=bool(rows[-1]["success30"])))
    report=dict(trace=str(a.trace.resolve()),trace_sha256=digest(a.trace),episodes=results,
                passed=all(r["qpos_max_error"]<1e-5 and r["audit_max_error"]<1e-4 and r["success_matches"] for r in results),
                scope="Same-backend batched reset, full integration state, independent native stepping of saved controls; saved previews only")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    if not report["passed"]:
        raise AssertionError("Saved trace differs from independent replay; investigate before reporting")


if __name__=="__main__":
    main()
