"""Freeze development-selected PPO models and their matched origins before test."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import numpy as np

from retarget_s09_reference import digest
from run_s09_study import ROOT, check_protocol
from summarize_s09_ppo_improvement import BASE, origin_and_budget

HELDOUT_START = 150000
HELDOUT_COUNT = 512


def selection_key(row):
    return (float(row["success30"]), float(row["hold_sec"]),
            -float(row["max_penetration_m"]), -int(float(row["step"])))


def terminal_checkpoint(run):
    """Fix the true endpoint independently of development checkpoint ranking."""
    run = run.resolve()
    config = json.loads((run / "run_config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] != "completed" or (config["eval_start"], config["eval_count"]) != (130000, 64):
        raise ValueError("Endpoint requires a completed development run")
    step = int(summary["final_step"])
    if step <= 0 or step != config["steps"]:
        raise ValueError("Endpoint must reach the declared training budget")
    with (run / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or int(float(rows[-1]["step"])) != step:
        raise ValueError("Endpoint summary disagrees with the last evaluation")
    checkpoint = run / f"step_{step:09d}/params.pkl"
    if not checkpoint.is_file():
        raise ValueError("Missing endpoint weights")
    return checkpoint, origin_and_budget(run, step)[1]


def require_zero_residual(params):
    head=params[1]["params"]["hidden_2"]
    kernel,bias=np.asarray(head["kernel"]),np.asarray(head["bias"])
    if kernel.shape[-1]!=34 or bias.shape!=(34,):
        raise ValueError("Expected a 17-action tanh-normal head")
    if not np.array_equal(kernel[:,:17],np.zeros_like(kernel[:,:17])) or not np.array_equal(bias[:17],np.zeros(17)):
        raise ValueError("Reference baseline must produce exactly zero correction")


def resolve_bc_checkpoint(checkpoint):
    """Separate extra critic fitting from the actual BC actor's origin/cost."""
    metadata_path=checkpoint.parent.parent/"critic_pretraining.json"
    if not metadata_path.exists():
        return checkpoint,None,None
    import jax
    metadata=json.loads(metadata_path.read_text())
    if metadata["status"]!="completed" or not metadata["actor_and_normalizer_exactly_unchanged"]:
        raise ValueError("Unverified value-only initialization")
    original=Path(metadata["source_checkpoint"])
    if digest(original)!=metadata["source_checkpoint_sha256"] or digest(checkpoint)!=metadata["checkpoint_sha256"]:
        raise ValueError("Changed critic initialization provenance")
    before=pickle.loads(original.read_bytes())[:2]
    after=pickle.loads(checkpoint.read_bytes())[:2]
    if jax.tree_util.tree_structure(before)!=jax.tree_util.tree_structure(after):
        raise ValueError("Actor/normalizer structure changed")
    for x,y in zip(jax.tree_util.tree_leaves(before),jax.tree_util.tree_leaves(after)):
        np.testing.assert_array_equal(x,y)
    recipe={key:metadata[key] for key in ("discount","collection_backend","collection_policy","collection_std",
        "collection_seed_range","training_episodes","validation_episodes","fit_updates","fit_batch_size","fit_learning_rate")}
    provenance=dict(metadata_path=str(metadata_path),metadata_sha256=digest(metadata_path),
        actual_environment_interactions=metadata["actual_environment_interactions"],
        source_bc_checkpoint=str(original),source_bc_checkpoint_sha256=digest(original))
    return original,recipe,provenance


def select_checkpoint(run):
    """Only inspect the declared development set and the restored ancestry."""
    candidates=[]
    recipe=[]
    current=run.resolve()
    seen=set()
    ceiling=None
    while True:
        if current in seen:
            raise ValueError("Cyclic checkpoint ancestry")
        seen.add(current)
        cfg=json.loads((current/"run_config.json").read_text())
        summary=json.loads((current/"summary.json").read_text())
        if summary["status"] != "completed" or (cfg["eval_start"],cfg["eval_count"]) != (130000,64):
            raise ValueError("Select only completed development runs")
        constraint_path=current/"constraint_config.json"
        constraint=json.loads(constraint_path.read_text()) if constraint_path.exists() else {}
        improvement_path=current/"improvement_config.json"
        improvement=json.loads(improvement_path.read_text()) if improvement_path.exists() else {}
        feedback_path=current/"feedback_config.json"
        feedback=json.loads(feedback_path.read_text()) if feedback_path.exists() else {}
        rehearsal_path=current/"rehearsal_config.json"
        rehearsal=json.loads(rehearsal_path.read_text()) if rehearsal_path.exists() else {}
        temporal_path=current/"temporal_config.json"
        temporal=json.loads(temporal_path.read_text()) if temporal_path.exists() else {}
        adapter_path=current/"adapter_config.json"
        adapter=json.loads(adapter_path.read_text()) if adapter_path.exists() else {}
        exploration_path=current/"exploration_config.json"
        exploration=json.loads(exploration_path.read_text()) if exploration_path.exists() else {}
        if adapter:
            audit_path=current/"adapter_audit.json"
            if not audit_path.exists():
                raise ValueError("Feedback-only run lacks its final frozen-weight audit")
            audit=json.loads(audit_path.read_text())
            if (audit.get("status")!="verified" or not audit.get("frozen_actor_parameters_exactly_unchanged")
                    or audit.get("final_checkpoint_sha256")!=digest(current/"params.pkl")
                    or audit.get("initial_checkpoint_sha256")!=digest(current/"step_000000000/params.pkl")):
                raise ValueError("Feedback-only weight audit is invalid or stale")
        recipe.append(dict(steps=cfg.get("steps"),evals=cfg.get("evals"),
            actor_lr=cfg.get("learning_rate"),critic_lr=constraint.get("critic_learning_rate") or cfg.get("learning_rate"),
            configured_std=cfg.get("std"),std_override=improvement.get("initial_std_override"),
            teacher_kl=constraint.get("teacher_kl_weight",0),
            discount=improvement.get("training_overrides",{}).get("discounting",.99),
            reward_version=cfg.get("env_config",{}).get("reward_version"),
            environment_config=cfg.get("env_config",{}),
            new_input_update_multiplier=feedback.get("new_input_update_multiplier",1.),
            demo_rehearsal={k:rehearsal.get(k) for k in ("dataset_sha256","demo_weight",
                "acquisition_steps","early_late_samples","full_feedback_inputs")},
            temporal_partition=dict(unroll_length=temporal.get("unroll_length",20),
                batch_size=temporal.get("batch_size",32)),
            actor_update_rule=adapter.get("actor_update_rule","all_actor_parameters"),
            phase_exploration={k:exploration.get(k) for k in ("acquisition_noise_scale",
                "phase_input_index","early_phase_upper_exclusive","minimum_effective_std")},
            transition_fix=(current/"transition_config.json").exists(),
            reward_event_penalty=cfg.get("env_config",{}).get("contact_break_penalty")))
        with (current/"metrics.csv").open() as f:
            for row in csv.DictReader(f):
                step=int(float(row["step"]))
                if ceiling is not None and step>ceiling:
                    continue
                origin,budget=origin_and_budget(current,step)
                # A zero-update BC/zero-residual policy is not an improved PPO.
                if budget:
                    candidates.append(dict(row=row,run=current,step=step,budget=budget,origin=origin))
        checkpoint=cfg.get("checkpoint")
        if not checkpoint:
            break
        checkpoint=Path(checkpoint).resolve()
        if digest(checkpoint)!=cfg["checkpoint_sha256"]:
            raise ValueError("Restored checkpoint changed since training")
        parent=checkpoint.parent.parent
        if not parent.is_relative_to(BASE):
            break
        ceiling=int(checkpoint.parent.name.removeprefix("step_"))
        current=parent
    if not candidates:
        raise ValueError("No trained candidate")
    # Budget breaks exact duplicate checkpoints across continuation boundaries.
    chosen=max(candidates,key=lambda c:(*selection_key(c["row"])[:3],-c["budget"]))
    if (chosen["run"]/"adapter_config.json").exists():
        from run_s09_feedback_adapter import verify_frozen_actor
        before=pickle.loads((chosen["run"]/"step_000000000/params.pkl").read_bytes())
        after=pickle.loads((chosen["run"]/f"step_{chosen['step']:09d}/params.pkl").read_bytes())
        verify_frozen_actor(before,after)
    chosen["declared_recipe"]=list(reversed(recipe))
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bc-runs",nargs=3,type=Path,required=True)
    p.add_argument("--residual-runs",nargs=3,type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reference-checkpoint",type=Path,
        default=BASE/"E11_residual_scaled_noise_lr/step_000000000/params.pkl")
    a=p.parse_args()
    check_protocol()
    if not a.output.resolve().is_relative_to(BASE):
        p.error("Keep new confirmation separate from frozen historical results")
    inputs=ROOT/"artifacts/orca_benchmark_suite/S09/inputs"
    jobs=[]
    selections=[]
    recipes={}

    def job(name,group,seed,checkpoint,conditioned):
        checkpoint=checkpoint.resolve() if checkpoint else None
        return dict(name=name,group=group,training_seed=seed,conditioned=conditioned,
            output=str((a.output.resolve()/"eval"/name).relative_to(ROOT)),
            method="reference" if checkpoint is None else "guided" if conditioned else "residual",
            entry="scripts/evaluate_s09_failure_quality.py",
            checkpoint=str(checkpoint.relative_to(ROOT)) if checkpoint else None,
            checkpoint_sha256=digest(checkpoint) if checkpoint else None)

    for group,runs in (("bc_ppo",a.bc_runs),("residual",a.residual_runs)):
        seeds=[]
        bc_hashes=[]
        for run in runs:
            chosen=select_checkpoint(run)
            endpoint,endpoint_budget=terminal_checkpoint(run)
            origin_cfg=json.loads((chosen["origin"]/"run_config.json").read_text())
            if origin_cfg["method"] != ("guided" if group=="bc_ppo" else "residual"):
                raise ValueError("Wrong method in confirmation group")
            bc,initialization,provenance=(resolve_bc_checkpoint(Path(origin_cfg["checkpoint"]))
                if group=="bc_ppo" else (None,None,None))
            recipe=dict(critic_initialization=initialization,stages=chosen["declared_recipe"])
            if group in recipes and recipes[group]!=recipe:
                raise ValueError("Independent repeats must share a recipe, not seed-specific hyperparameter winners")
            recipes[group]=recipe
            seed=origin_cfg["seed"]
            seeds.append(seed)
            checkpoint=chosen["run"]/f"step_{chosen['step']:09d}/params.pkl"
            jobs.append(job(f"{group}_s{seed}",group,seed,checkpoint,group=="bc_ppo"))
            if group=="bc_ppo":
                bc_cfg=json.loads((bc.parent.parent/"run_config.json").read_text())
                if bc_cfg["steps"]!=0 or bc.parent.name!="step_000000000":
                    raise ValueError("BC origin must precede online training")
                bc_hashes.append(digest(bc))
                jobs[-1]["bc_pretraining_seed"]=bc_cfg["seed"]
                baseline_job=job(f"bc_s{seed}","bc",seed,bc,True)
                baseline_job["bc_pretraining_seed"]=bc_cfg["seed"]
                jobs.append(baseline_job)
                last_job=job(f"bc_ppo_last_s{seed}","bc_ppo_last",seed,endpoint,True)
                last_job["bc_pretraining_seed"]=bc_cfg["seed"]
                jobs.append(last_job)
            selections.append(dict(group=group,training_seed=seed,
                requested_completed_run=str(run.resolve().relative_to(ROOT)),
                selected_checkpoint=str(checkpoint.relative_to(ROOT)),
                terminal_checkpoint=str(endpoint.relative_to(ROOT)),
                terminal_checkpoint_sha256=digest(endpoint),
                terminal_model_path_online_steps=endpoint_budget,
                selected_is_terminal=digest(checkpoint)==digest(endpoint),
                selected_model_path_online_steps=chosen["budget"],
                selected_model_path_total_environment_interactions=chosen["budget"]+(provenance["actual_environment_interactions"] if provenance else 0),
                extra_critic_pretraining=provenance,
                development_row=chosen["row"]))
        if len(set(seeds))!=3:
            raise ValueError("Require three independent training origins, not three continuations of one model")
        if group=="bc_ppo" and len(set(bc_hashes))!=3:
            raise ValueError("Require three distinct BC origins")
    reference_config=json.loads((a.reference_checkpoint.parent.parent/"run_config.json").read_text())
    if reference_config["method"]!="residual":
        raise ValueError("Use the same base-plus-correction control path for the zero baseline")
    require_zero_residual(pickle.loads(a.reference_checkpoint.read_bytes()))
    jobs.append(job("reference","reference",0,a.reference_checkpoint,False))
    sources=set((ROOT/"scripts").glob("*s09*.py"))
    sources.update(ROOT/"scripts"/name for name in (
        "orca_native_physics.py","orca_contact_forces.py","orca_matched_env.py",
        "train_orca_comparison.py","train_orca_matched.py"))
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input=str(inputs.relative_to(ROOT)),eval_start=HELDOUT_START,eval_count=HELDOUT_COUNT,
        evaluation_design="Expanded from 128 to 512 before any heldout execution. Development paired variance 0.04340278 and difference 0.02604167 suggest approximately 502 cases by a normal approximation; this excludes training-seed uncertainty and guarantees neither power nor improvement.",
        development_range=[130000,130063],
        selection="Completed declared ancestry only; success, hold, lower depth, earlier cumulative step. No test selection.",
        endpoint_diagnostic="Also evaluate all three BC+PPO final endpoints regardless of development score. These are the same training runs, not additional independent repeats. Never choose between selected and terminal models using heldout results.",
        model_sha256=digest(inputs/"model.xml"),reference_sha256=digest(inputs/"reference.npz"),
        dataset_sha256=digest(inputs.parent/"demonstrations/trajectories.npz"),
        criterion="Unchanged S09 strict released continuous30s, final valid; failures retained",
        reference_baseline="Exact zero mean residual network through the same base-plus-correction control path; no inverse normalization roundtrip",
        training_seed_semantics="PPO-origin seed, reused as the paired BC identifier; actual BC training seed is bc_pretraining_seed. Reference 0 is an identifier, not a training run.",
        limitations="Development-selected models; unequal search and path budgets; three training repeats only; not a pure-PPO budget benchmark",
        execution_sources={str(f.relative_to(ROOT)):digest(f) for f in sorted(sources)},
        declared_recipes=recipes,selections=selections,jobs=jobs)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"plan.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(plan=str(a.output/"plan.json"),sha256=digest(a.output/"plan.json"),jobs=len(jobs))))


if __name__ == "__main__":
    main()
