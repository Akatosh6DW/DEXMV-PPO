"""Audit checkpoint updates and every saved S12 evaluation action/physics trace."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import mujoco
import numpy as np

from prepare_s12_evaluation import validate_source


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(tree):
    if isinstance(tree, dict):
        return [a for key in sorted(tree) for a in arrays(tree[key])]
    if isinstance(tree, (tuple,list)):
        return [a for value in tree for a in arrays(value)]
    return [np.asarray(tree)]


def delta(a,b):
    left,right = arrays(a),arrays(b)
    if len(left)!=len(right):
        raise ValueError("Parameter tree structure differs")
    return float(np.sqrt(sum(np.square(x.astype(float)-y.astype(float)).sum() for x,y in zip(left,right))))


def residual_effect(trace, low, high):
    base = np.clip(trace["video_base"],low,high)
    previous = np.vstack([np.zeros_like(trace["ctrl"][:1]),trace["ctrl"][:-1]])
    without_residual = np.clip(base,previous-.1,previous+.1)
    rms = lambda x:float(np.sqrt(np.mean(np.square(x))))
    return dict(requested_rms_mean_rad=rms(trace["requested_residual"]),
        bounded_target_correction_rms_mean_rad=rms(trace["desired_ctrl"]-base),
        one_step_command_effect_rms_mean_rad=rms(trace["ctrl"]-without_residual),
        max_requested_rad=float(np.abs(trace["requested_residual"]).max()))


def physical_retention(trace, config):
    """Recompute the relaxed gate, which still requires paired designated pads."""
    valid = ((trace["time"] >= config["release_sec"] + .5)
        & trace["alive"].astype(bool) & (trace["vertices"].mean(1)[:, 2] >= .535)
        & (trace["support_force_n"] < 1e-5) & (trace["loads"][:, 0] > .02)
        & np.any(trace["loads"][:, 1:] > .02, axis=1))
    hold = best = 0
    for value in valid:
        hold = hold + 1 if value else 0
        best = max(best, hold)
    return bool(hold * config["control_dt"] >= config["hold_sec"]), best * config["control_dt"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.study/"protocol.json").read_text())
    ep = json.loads((a.study/"evaluation_plan.json").read_text())
    if ep["protocol_sha256"]!=digest(a.study/"protocol.json"):
        raise ValueError("Protocol changed")
    root = Path(__file__).resolve().parents[1]
    for name, expected in plan["inputs"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    updates = []
    for job in plan["jobs"]:
        run = Path(job["output"])
        validate_source(json.loads((run/"run_config.json").read_text()),job,plan)
        first = pickle.loads((run/"step_000000000/params.pkl").read_bytes())
        last = pickle.loads((run/f"step_{plan['online_steps']:09d}/params.pkl").read_bytes())
        actor,critic = delta(first[1],last[1]),delta(first[2],last[2])
        if actor<=1e-9 or critic<=1e-9:
            raise ValueError("No real actor/critic updates: "+job["name"])
        if job["method"]=="guided":
            bc = pickle.loads((run/"bc_params.pkl").read_bytes())
            if delta(bc[1],first[1])>1e-10:
                raise ValueError("BC checkpoint differs from PPO initialization")
            kernel = np.asarray(bc[1]["params"]["hidden_0"]["kernel"])
            mask = np.ones(93,bool); mask[87:90] = False
            if np.any(kernel[mask]!=0):
                raise ValueError("Phase-only BC input mask not preserved")
        updates.append(dict(run=job["name"],actor_l2_change=actor,critic_l2_change=critic))
    audits, physics_failures = [], []
    for job in ep["jobs"]:
        run = Path(job["output"])
        summary = json.loads((run/"summary.json").read_text())
        records = json.loads((run/"episodes.json").read_text())
        config = json.loads((run/"evaluation_config.json").read_text())
        if len(records)!=job["eval_count"] or summary["status"]!="completed":
            raise ValueError("Missing evaluation cases")
        if ([r["seed"] for r in records] != plan["conditions"][job["condition"]]["seeds"]
                or config["condition"] != job["condition"]
                or config["checkpoint_sha256"] != job["checkpoint_sha256"]
                or digest(run / "model.xml") != config["model_sha256"]):
            raise ValueError("Evaluation provenance mismatch")
        if job["checkpoint"] and digest(job["checkpoint"])!=job["checkpoint_sha256"]:
            raise ValueError("Bound checkpoint changed")
        model = mujoco.MjModel.from_xml_path(str(run/"model.xml"))
        if model.nu!=17 or model.nflex!=1 or model.neq!=1:
            raise ValueError("Unexpected object actuator or constraint structure")
        low,high = model.actuator_ctrlrange.T
        maximum_error = 0.
        effects = []
        for record in records:
            with np.load(run/f"trajectory_{record['seed']}.npz") as trace:
                if not all(np.isfinite(trace[k]).all() for k in trace.files):
                    raise ValueError("Nonfinite trace")
                if job["label"]=="residual":
                    np.testing.assert_allclose(trace["requested_residual"],.08*trace["action"],atol=2e-7)
                    desired = np.clip(trace["video_base"]+trace["requested_residual"],low,high)
                    if np.max(np.abs(trace["requested_residual"]))>.080001:
                        raise ValueError("Residual limit violated")
                else:
                    if np.any(trace["video_base"]!=0) or np.any(trace["requested_residual"]!=0):
                        raise ValueError("Unexpected execution-time reference in absolute-action policy")
                    desired = low+(trace["action"]+1)*.5*(high-low)
                error = float(np.abs(desired-trace["desired_ctrl"]).max())
                maximum_error = max(maximum_error,error)
                np.testing.assert_allclose(desired,trace["desired_ctrl"],atol=2e-6,rtol=0)
                previous = np.vstack([np.zeros((1,17)),trace["ctrl"][:-1]])
                executed = np.clip(np.clip(desired,low,high),previous-.1,previous+.1)
                np.testing.assert_allclose(executed,trace["ctrl"],atol=2e-6,rtol=0)
                if job["label"]=="residual":
                    effects.append(residual_effect(trace,low,high))
                if bool(trace["success15"][-1])!=record["success15"]:
                    raise ValueError("Episode outcome and trace disagree")
                physical, best_physical = physical_retention(trace, config["soft_config"])
                if physical != record["physical_success15"] or not np.isclose(
                        best_physical, record["physical_best_hold_sec"], atol=1e-8, rtol=0):
                    raise ValueError("Physical retention metric and trace disagree")
                if record["success15"]:
                    if len(trace["time"])!=480 or trace["hold_sec"][-1]<15 or trace["support_force_n"][-1]>1e-5:
                        raise ValueError("Success without full unsupported hold")
                minimum_volume = float(trace["min_volume_ratio"].min())
                if minimum_volume<=.05:
                    first = int(np.flatnonzero(trace["min_volume_ratio"]<=.05)[0])
                    if record["success15"] or bool(trace["alive"][-1]) or first!=len(trace["time"])-1:
                        raise ValueError("Invalid element was not terminated and counted as failure")
                    physics_failures.append(dict(job=job["name"],seed=record["seed"],
                        first_time_sec=float(trace["time"][first]),min_volume_ratio=minimum_volume,
                        inverted=minimum_volume<=0,trace=str(run/f"trajectory_{record['seed']}.npz")))
        row = dict(job=job["name"],method=job["label"],condition=job["condition"],
            train_seed=job["train_seed"],count=len(records),action_max_error_rad=maximum_error)
        if effects:
            row.update({k:float((max if k=="max_requested_rad" else np.mean)([r[k] for r in effects]))
                        for k in effects[0]})
        audits.append(row)
    output = a.study/"report/verification.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(status="completed_with_physics_failures" if physics_failures else "passed",
        data_integrity="passed",all_episodes_physically_valid=not physics_failures,
        physics_failures=physics_failures,protocol_sha256=digest(a.study/"protocol.json"),
        training_updates=updates,evaluation_audits=audits,total_episodes=sum(x["count"] for x in audits)),indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    main()
