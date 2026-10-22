"""Same pre-existing candidate offsets, short versus full-horizon ranking.

This diagnoses the recovery teacher on training states, not policy performance.
No new search, student updates, or held-out cases enter candidate selection.
"""

import argparse
import json
from pathlib import Path
import pickle
import shutil

import jax
import jax.numpy as jp
import numpy as np

from analyze_s09_bc_confirmation import write_csv
from collect_s09_student_corrections import canonical_plan
from probe_s09_recovery_teacher import ROOT, unpack_counters, offset_controls, rollout, objective_value
from probe_s09_recovery_timing import phase_targets, verify_replay
from retarget_s09_reference import digest
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_bc_phase_curriculum import KEEP
from s09_suspension import SuspendedCylinder
import run_s09_job as runner


def shortlist(history, top=8):
    if not history or top < 1 or any(not np.isfinite(r["cost"]) for r in history):
        raise ValueError("Require finite recorded candidate costs and positive top count")
    ranked = sorted(range(len(history)), key=lambda i: (history[i]["cost"],i))
    indices, seen = [], set()
    for i in ranked:
        delta = tuple(history[i]["delta"])
        if len(delta)!=10 or not np.isfinite(delta).all() or np.max(np.abs(delta)) > .0800000001:
            raise ValueError("Invalid historical bounded candidate")
        if delta not in seen:
            indices.append(i)
            seen.add(delta)
        if len(indices)==top:
            break
    zero = next(i for i,r in enumerate(history) if not np.any(r["delta"]))
    if tuple(history[zero]["delta"]) not in seen:
        indices.append(zero)
    return indices


def select_cases(available, requested, smoke):
    chosen=list(available) if requested is None else list(requested)
    if not chosen or len(chosen)!=len(set(chosen)) or not set(chosen).issubset(available):
        raise ValueError("Only unique historical training queries are allowed")
    return chosen[:1] if smoke else chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--candidate-count",type=int,default=8)
    p.add_argument("--query-seeds",type=int,nargs="+")
    a = p.parse_args()
    if not 1 <= a.candidate_count <= 161:
        p.error("Candidate count must lie in [1,161]")
    assert jax.default_backend()=="cpu"
    jax.config.update("jax_default_matmul_precision","highest")
    origin = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018/D38_student_correction_round"
    read = lambda path:json.loads(path.read_text())
    source = read(origin/"protocol.json")
    queries = read(origin/"queries.json")
    assert not source["smoke"] and source["prefix_steps"]==40 and source["search_horizon_steps"]==160
    assert set(queries).issubset(source["training_seeds"])
    model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
    nominal = Path(source["nominal"])
    assert digest(model)==source["model_sha256"] and digest(nominal)==source["nominal_sha256"]
    cfg = read(nominal.parent.parent/"run_config.json")
    assert digest(model.parent/"reference.npz")==cfg["reference_sha256"]
    for name in ("probe_s09_recovery_teacher.py","probe_s09_recovery_timing.py","s09_suspension.py"):
        assert digest(ROOT/"scripts"/name)==digest(origin/"sources"/name)
    env = ModeConditionedGrasp(model,"pure")
    net,_ = runner.initialize_matched(env,cfg["seed"],cfg["std"])
    params = pickle.loads(nominal.read_bytes())
    np.testing.assert_array_equal(np.asarray(params[1]["params"]["hidden_0"]["kernel"])[~KEEP],0)
    policy = runner.networks.make_inference_fn(net)(params,deterministic=True)
    infer = jax.jit(lambda obs:policy(dict(state=obs),jax.random.PRNGKey(0))[0])
    a.output.mkdir(parents=True,exist_ok=False)
    snapshots = a.output/"sources"
    snapshots.mkdir()
    sources={}
    import sys
    for module in list(sys.modules.values()):
        file = getattr(module,"__file__",None)
        if file and Path(file).resolve().parent==ROOT/"scripts" and file.endswith(".py"):
            file=Path(file).resolve()
            shutil.copyfile(file,snapshots/file.name)
            sources[file.name]=digest(file)
    cases = select_cases(queries,a.query_seeds,a.smoke)
    top=1 if a.smoke else a.candidate_count
    protocol=dict(origin=str(origin),origin_sha256=digest(origin/"protocol.json"),training_seeds=cases,
        candidate_rule=f"top {top} unique existing short-cost candidates plus zero",candidate_count_requested=top,
        no_new_search=True,short_horizon_steps=160,long_horizon_steps=760,smoke=a.smoke,
        selection="argmin of the SAME objective formula, only horizon changes; strict-success oracle reported separately",
        model_sha256=digest(model),checkpoint_sha256=digest(nominal),sources=sources,policy_updates=0,
        limitation="Selected old training failures and short-cost shortlist, not exhaustive search, new video, or policy generalization")
    (a.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    original_rows={r["seed"]:r for r in read(origin/"collection.json")}
    records, all_candidates, costs=[],[],dict(full_horizon_selection=0,selected_replay_validation=0,
        canonical_collection=0,canonical_validation=0)
    for seed in cases:
        folder=a.output/f"seed{seed}"
        folder.mkdir()
        old=origin/f"seed{seed}"
        history=read(old/"search.json")
        indices=shortlist(history,top)
        state=jax.device_get(jax.jit(env.reset)(jax.random.PRNGKey(seed)))
        sim=SuspendedCylinder(model.read_text())
        desired=phase_targets(state,infer,sim.model.actuator_ctrlrange).astype(float)
        with np.load(old/"initial_state.npz") as z:
            snapshot=(z["state"].copy(),unpack_counters(z["spawn"],z["counters"]))
        with np.load(old/"nominal_zero.npz") as z:
            np.testing.assert_array_equal(desired[40:40+len(z["desired_ctrl"])],z["desired_ctrl"])
        candidate_rows, traces = [], {}
        for i in indices:
            delta=np.array(history[i]["delta"])
            controls=offset_controls(desired[40:],delta,np.arange(7,17),sim.model.actuator_ctrlrange)
            result,trace=rollout(sim,snapshot,controls)
            costs["full_horizon_selection"]+=len(trace["alive"])
            row=dict(seed=seed,index=i,short_cost=history[i]["cost"],long_cost=objective_value(result,delta),
                success30=bool(trace["success30"][-1] and trace["alive"][-1]),
                best_hold_sec=float(trace["best_hold_sec"].max()),delta=delta.tolist(),**result)
            candidate_rows.append(row)
            traces[i]=trace
            np.savez_compressed(folder/f"candidate_{i:03d}.npz",**trace)
        short=min(candidate_rows,key=lambda r:r["short_cost"])
        long=min(candidate_rows,key=lambda r:r["long_cost"])
        np.testing.assert_array_equal(short["delta"],original_rows[seed]["selected_delta"])
        assert short["success30"]==original_rows[seed]["selected_offset"]["success30"]
        with np.load(old/"selected_offset.npz") as z:
            for key in z.files:
                np.testing.assert_array_equal(traces[short["index"]][key],z[key])
        for i in {short["index"],long["index"]}:
            controls=offset_controls(desired[40:],np.array(history[i]["delta"]),np.arange(7,17),sim.model.actuator_ctrlrange)
            _,repeat=rollout(sim,snapshot,controls)
            costs["selected_replay_validation"]+=len(repeat["alive"])
            for key in repeat:
                np.testing.assert_array_equal(repeat[key],traces[i][key])
        canonical=None
        if long["success30"]:
            with np.load(origin/"student_rollouts.npz") as z:
                index=list(z["seeds"]).index(seed)
                desired[:40]=z["desired_ctrl"][index,:40]
            desired[40:]=offset_controls(desired[40:],np.array(long["delta"]),np.arange(7,17),sim.model.actuator_ctrlrange)
            canonical,collected=canonical_plan(env,net,seed,desired,folder,model)
            costs["canonical_collection"]+=int(collected["active"].sum())
            costs["canonical_validation"]+=int(collected["active"].sum())
        record=dict(seed=seed,candidates=len(indices),short_index=short["index"],long_index=long["index"],
            short_success=short["success30"],long_success=long["success30"],
            short_hold_sec=short["best_hold_sec"],long_hold_sec=long["best_hold_sec"],
            any_candidate_success=any(r["success30"] for r in candidate_rows),canonical=canonical,
            short_trace_matches_original=True,selected_replay_identical=True,
            search_sha256=digest(old/"search.json"),snapshot_sha256=digest(old/"initial_state.npz"))
        records.append(record)
        all_candidates+=candidate_rows
        (folder/"candidates.json").write_text(json.dumps(candidate_rows,indent=2)+"\n")
        (a.output/"results.json").write_text(json.dumps(records,indent=2)+"\n")
        print(json.dumps(record),flush=True)
    write_csv(a.output/"candidates.csv",all_candidates)
    summary=dict(status="completed",smoke=a.smoke,cases=len(cases),short_successes=sum(r["short_success"] for r in records),
        long_successes=sum(r["long_success"] for r in records),
        any_candidate_successes=sum(r["any_candidate_success"] for r in records),
        canonical_successes=sum(bool(r["canonical"] and r["canonical"]["qualified"]) for r in records),
        costs=costs,policy_updates=0,new_search_evaluations=0,
        limitation=protocol["limitation"],source_sha256=digest(Path(__file__)))
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
