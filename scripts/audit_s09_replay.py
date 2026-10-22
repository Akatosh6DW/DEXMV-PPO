"""Replay saved S09 desired controls in native MuJoCo, outside the RL step loop."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Aero-Hand/sim_rl/mujoco_playground"))

import jax
import mujoco
import numpy as np

from orca_native_physics import STATE_SPEC
from orca_s09_env import OrcaS09Grasp
from retarget_s09_reference import digest
from s09_suspension import SuspendedCylinder


def check_residual_composition(trace, index, limits):
    active = trace["active"][index].astype(bool)
    assert active.any()
    expected = np.clip(trace["video_base"][index] + .08*np.clip(trace["action"][index], -1, 1), *limits.T)
    np.testing.assert_allclose(trace["desired_ctrl"][index, active], expected[active], atol=2e-7)
    np.testing.assert_allclose(trace["requested_residual"][index, active],
                               .08*np.clip(trace["action"][index, active], -1, 1), atol=2e-7)
    assert np.max(np.abs(trace["requested_residual"][index, active])) <= .080001
    assert np.max(np.abs(trace["effective_residual"][index, active])) <= .08001
    # The evaluator still computes unused policy outputs after termination.
    # Its physical/control state must be frozen, not reconstructed from those outputs.
    last = int(np.flatnonzero(active)[-1])
    np.testing.assert_array_equal(active, np.arange(len(active)) <= last)
    for key in ("qpos", "qvel", "ctrl", "desired_ctrl", "video_base",
                "requested_residual", "effective_residual", "audit"):
        values = trace[key][index]
        np.testing.assert_array_equal(values[~active], np.broadcast_to(values[last], values[~active].shape))
    return float(np.max(np.abs(trace["desired_ctrl"][index, active] - expected[active])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--step",type=int,default=0)
    a=p.parse_args()
    config=json.loads((a.run/"run_config.json").read_text())
    if jax.default_backend()!=config["backend"]:
        raise ValueError("Replay reset must use the recorded backend for exact RNG values")
    inputs=Path(config["input"])
    model,bank=inputs/"model.xml",inputs/"reference.npz"
    assert digest(model)==config["model_sha256"] and digest(bank)==config["reference_sha256"]
    method=config["method"]
    env=OrcaS09Grasp(model,"residual" if method=="residual" else "pure",
        bank if method in ("residual","reference","collect") else None,
        approach_weight=config["approach_weight"],coverage_weight=config["coverage_weight"],
        orientation_mode=config["orientation"])
    reset=jax.jit(env.reset)
    folder=a.run/f"step_{a.step:09d}"
    with np.load(folder/"preview_trajectories.npz") as z:
        trace=dict(z)
    recorded={r["seed"]:r for r in json.loads((folder/"episodes.json").read_text())}
    rows=[]
    for i,seed in enumerate(trace["seeds"]):
        state=jax.device_get(reset(jax.random.PRNGKey(int(seed))))
        sim=SuspendedCylinder(model.read_text())
        sim.reset(np.asarray(state.info["cube_spawn_pos"]),np.asarray(state.data.qpos[env.oid+3:env.oid+7]))
        mujoco.mj_setState(sim.model,sim.data,np.asarray(state.data.integration_words).copy().view(np.float64),STATE_SPEC)
        max_q=max_v=max_audit=0.
        controls=trace["desired_ctrl"][i]
        composition_error = None
        if method=="residual":
            composition_error = check_residual_composition(trace, i, sim.model.actuator_ctrlrange)
        else:
            np.testing.assert_array_equal(trace["obs"][i,:,95:112],0)
        active=np.flatnonzero(trace["active"][i])
        for k in active:
            row=sim.step(controls[k])
            max_q=max(max_q,float(np.max(np.abs(trace["qpos"][i,k]-sim.data.qpos))))
            max_v=max(max_v,float(np.max(np.abs(trace["qvel"][i,k]-sim.data.qvel))))
            packed=np.r_[row["loads"],row["forbidden"],row["penetration_m"],row["peak_force_n"],
                         row["rope_force_n"],row["alive"],row["released"],row["valid"],row["time"]]
            max_audit=max(max_audit,float(np.max(np.abs(trace["audit"][i,k]-packed))))
            assert bool(trace["alive"][i,k])==row["alive"]
            np.testing.assert_allclose(trace["hold"][i,k],row["hold_sec"] if row["alive"] else 0,atol=5e-6)
        episode=recorded[int(seed)]
        assert episode["success30"]==row["success30"]
        assert max_q<3e-6 and max_v<1e-4 and max_audit<1e-4,(seed,max_q,max_v,max_audit)
        rows.append(dict(seed=int(seed),steps=len(active),success30=row["success30"],
            max_qpos_error=max_q,max_qvel_error=max_v,max_audit_error=max_audit,passed=True,
            active_composition_error=composition_error, frozen_padding_steps=len(controls)-len(active)))
    result=dict(status="verified",run=str(a.run),step=a.step,episodes=rows,
        trace_sha256=digest(folder/"preview_trajectories.npz"),
        scope="Every executed step in saved previews; post-terminal padding is not an executed action; direct native replay, not all unrecorded episodes; shared rope/contact definitions remain explicit")
    (folder/"native_replay_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
