"""Precommit the existing historical-actor selection rule across three lineages."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import jax
import jax.numpy as jp
import numpy as np

from confirm_s09_experts_multisource import OUT as PREVIOUS, SOURCES
from make_s09_checkpoint_portfolio import choose_sources
from retarget_s09_reference import digest
from run_s09_mode_experts import ARCHITECTURE, BASE_FACTORY, network_factory

ROOT = Path(__file__).resolve().parents[1]
BASE = PREVIOUS.parent
OUT = BASE / "confirmation_portfolio_multisource_20261021"
START, COUNT = 200000, 256
GROUPS = ("bc", "stage", "expert_last", "portfolio")


def read(path):
    return json.loads(path.read_text())


def validate_selection_rows(rows):
    if len(rows) != 64 or [r["seed"] for r in rows] != list(range(130000, 130064)):
        raise ValueError("Selection must use the original ordered 64 development states")
    if any(r["selected_reference"] not in range(6) or type(r["success30"]) is not bool for r in rows):
        raise ValueError("Invalid development label or routing")
    return [sum(r["success30"] for r in rows if r["selected_reference"] == m) for m in range(6)]


def assemble(source, previous_jobs, bc_run):
    paths = {g: ROOT / previous_jobs[(source, g)]["checkpoint"] for g in GROUPS[:-1]}
    folders = [BASE / bc_run / "step_000000000", paths["stage"].parent, paths["expert_last"].parent]
    configs = [read(p.parent / "run_config.json") for p in folders]
    parameters, counts, provenance = [], [], []
    initial = None
    for group, folder, cfg in zip(GROUPS[:-1], folders, configs):
        assert cfg["eval_start"] == 130000 and cfg["eval_count"] == 64
        for key in ("model_sha256", "reference_sha256", "env_config"):
            assert cfg[key] == configs[0][key]
        with np.load(folder / "initial_states.npz") as z:
            np.testing.assert_array_equal(z["seeds"], np.arange(130000, 130064))
            if initial is None:
                initial = z["qpos"].copy()
            np.testing.assert_array_equal(z["qpos"], initial)
        counts.append(validate_selection_rows(read(folder / "episodes.json")))
        parameters.append(pickle.loads((folder / "params.pkl").read_bytes()))
        provenance.append(dict(group=group, checkpoint=str((folder / "params.pkl").relative_to(ROOT)),
            checkpoint_sha256=digest(folder / "params.pkl"), episodes_sha256=digest(folder / "episodes.json")))
    original_bc = pickle.loads(paths["bc"].read_bytes())
    for x, y in zip(jax.tree_util.tree_leaves(parameters[0][1]), jax.tree_util.tree_leaves(original_bc[1])):
        np.testing.assert_array_equal(x, y)
    choices = choose_sources(counts)
    actors = tuple(parameters[k][1]["experts"][m] if k == 2 else parameters[k][1] for m, k in enumerate(choices))
    combined = (parameters[2][0], {"experts": actors}, parameters[2][2])
    base = BASE_FACTORY(dict(state=(123,), privileged_state=(140,)), 17)
    net = network_factory(dict(state=(123,), privileged_state=(140,)), 17)
    errors = []
    for mode, k in enumerate(choices):
        with np.load(folders[k] / "preview_trajectories.npz") as z:
            obs = z["obs"][z["active"]].copy()
        obs[:, 95:101] = 0
        obs[:, 95+mode] = 1
        inputs = dict(state=jp.asarray(obs))
        a = np.asarray(base.parametric_action_distribution.mode(base.policy_network.apply(parameters[k][0], actors[mode], inputs)))
        b = np.asarray(net.parametric_action_distribution.mode(net.policy_network.apply(combined[0], combined[1], inputs)))
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
        errors.append(float(np.max(np.abs(a-b))))
    if source == 14122:
        old = pickle.loads((BASE / "confirmation_portfolio_20261019/candidate/params.pkl").read_bytes())
        assert jax.tree_util.tree_structure(old) == jax.tree_util.tree_structure(combined)
        for x, y in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(combined)):
            np.testing.assert_array_equal(x, y)
    selection = dict(status="assembled_not_newly_trained", training_source=source, sources=provenance,
        development_seeds=[130000, 130063], source_success_counts=counts,
        selected_source_by_mode=[GROUPS[k] for k in choices], tie_rule="BC then stage then endpoint",
        predicted_development_successes=sum(counts[k][m] for m, k in enumerate(choices)),
        predicted_not_yet_executed=True, max_same_input_action_errors=errors,
        extra_ppo_steps=0, inherited_path_online_steps=1310720, actor_parameter_multiplier=6,
        source_14122_identical_to_old_candidate=source == 14122,
        normalization="Raw observations; accumulated source statistics are not used by the policy factory")
    return combined, configs[2], selection


def prepare():
    assert not OUT.exists(), "Do not overwrite an existing plan or partial preparation"
    old_plan = read(PREVIOUS / "plan.json")
    assert read(PREVIOUS / "report/statistics.json")["status"] == "completed_precommitted_confirmation"
    for path, sha in old_plan["execution_sources"].items():
        assert digest(ROOT / path) == sha, path
    scanned = 0
    for path in (ROOT / "artifacts/orca_benchmark_suite/S09").rglob("*.json"):
        if path.name not in ("run_config.json", "plan.json", "protocol.json"):
            continue
        cfg = read(path)
        start, count = cfg.get("eval_start"), cfg.get("eval_count")
        if isinstance(start, int) and isinstance(count, int):
            scanned += 1
            assert not (start < START+COUNT and start+count > START), str(path)
    jobs_by_key = {(j["training_source"], j["group"]): j for j in old_plan["jobs"]}
    jax.config.update("jax_default_matmul_precision", "highest")
    candidates = [assemble(source, jobs_by_key, bc_run) for (source, _), bc_run in zip(SOURCES,
        ("E19_bc_fast_critic_long", "E27_bc_fast_critic_s14102", "E29_bc_fast_critic_s14103"))]
    OUT.mkdir()
    jobs, selections = [], {}
    for (source, _), (params, cfg, selection) in zip(SOURCES, candidates):
        candidate = OUT / "candidates" / f"s{source}"
        candidate.mkdir(parents=True)
        checkpoint = candidate / "params.pkl"
        checkpoint.write_bytes(pickle.dumps(jax.device_get(params)))
        compatible = dict(cfg, steps=0, policy_architecture=ARCHITECTURE,
            configuration_role="Inference compatibility only; not a new training run",
            policy_selection="per-pose historical actors selected on old development states", extra_training_steps=0)
        (candidate / "run_config.json").write_text(json.dumps(compatible, indent=2) + "\n")
        selection["candidate_sha256"] = digest(checkpoint)
        (candidate / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        selections[str(source)] = dict(path=str((candidate / "selection.json").relative_to(ROOT)),
            sha256=digest(candidate / "selection.json"))
        for group in GROUPS:
            if group == "portfolio":
                path, multiplier, budget = checkpoint, 6, 1310720
                source_cfg = candidate / "run_config.json"
                seed = source
            else:
                old = jobs_by_key[(source, group)]
                path, multiplier, budget = ROOT / old["checkpoint"], old["actor_parameter_multiplier"], old["path_online_steps"]
                source_cfg = path.parent.parent / "run_config.json"
                seed = old["training_seed"]
            name = f"{group}_s{source}"
            jobs.append(dict(name=name, group=group, training_source=source, training_seed=seed,
                checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=digest(path),
                source_config=str(source_cfg.relative_to(ROOT)), source_config_sha256=digest(source_cfg),
                actor_parameter_multiplier=multiplier, path_online_steps=budget,
                output=str((OUT / "eval" / name).relative_to(ROOT)),
                entry="scripts/evaluate_s09_mode_expert_quality.py" if multiplier == 6 else "scripts/evaluate_s09_failure_quality.py",
                auditor="scripts/audit_s09_mode_experts.py" if multiplier == 6 else "scripts/audit_s09_bc_repair.py"))
    sources = dict(old_plan["execution_sources"])
    for name in (Path(__file__).name, "make_s09_checkpoint_portfolio.py"):
        sources[f"scripts/{name}"] = digest(ROOT / "scripts" / name)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), eval_start=START, eval_count=COUNT,
        input=old_plan["input"], model_sha256=old_plan["model_sha256"], reference_sha256=old_plan["reference_sha256"],
        env_config=old_plan["env_config"], jobs=jobs, selections=selections, execution_sources=sources,
        scanned_records=scanned, prior_confirmation_sha256=digest(PREVIOUS / "plan.json"),
        primary_contrast="portfolio minus bc", secondary_contrasts=["portfolio minus stage", "portfolio minus expert_last"],
        selection="Only original 64 development labels; fixed old BC/stage/endpoint rule, earliest ties; all three lineages",
        statistics="Paired lineage/common-state descriptive bootstrap 20000 draws; three training lineages, not 768 independent trainings",
        extra_ppo_steps=0, limitations="Historical checkpoint assembly, not a newly trained PPO or residual; sixfold actor capacity; no hardware safety claim")
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(plan_sha256=digest(OUT / "plan.json"), selection=[s for _, _, s in candidates])), flush=True)


def execute(name):
    plan = read(OUT / "plan.json")
    assert (plan["eval_start"], plan["eval_count"]) == (START, COUNT)
    for path, sha in plan["execution_sources"].items():
        assert digest(ROOT / path) == sha, path
    for item in plan["selections"].values():
        assert digest(ROOT / item["path"]) == item["sha256"]
    for filename, key in (("model.xml", "model_sha256"), ("reference.npz", "reference_sha256")):
        assert digest(ROOT / plan["input"] / filename) == plan[key]
    job = next(j for j in plan["jobs"] if j["name"] == name)
    assert digest(ROOT / job["checkpoint"]) == job["checkpoint_sha256"]
    assert digest(ROOT / job["source_config"]) == job["source_config_sha256"]
    logs = OUT / "execution"
    logs.mkdir(exist_ok=True)
    lock = (logs / f"{name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = ROOT / job["output"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_ENABLE_COMPILATION_CACHE="false")
    if output.exists() and not (output / "summary.json").exists():
        raise RuntimeError("Inspect partial output and live processes; do not overwrite")
    if not output.exists():
        assert shutil.disk_usage(ROOT).free > 1024**3
        command = [sys.executable, str(ROOT / job["entry"]), "--input", plan["input"], "--output", str(output),
            "--method", "guided", "--steps", "0", "--checkpoint", str(ROOT / job["checkpoint"]),
            "--eval-start", str(START), "--eval-count", str(COUNT), "--seed", str(job["training_seed"]), "--std", ".03"]
        print(json.dumps(dict(status="start", name=name)), flush=True)
        with (logs / f"{name}.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    summary = read(output / "summary.json")
    assert summary["status"] == "completed" and summary["final_step"] == 0
    audit_path = output / "step_000000000/native_replay_audit.json"
    if not audit_path.exists():
        with (logs / f"{name}_audit.log").open("x") as stream:
            subprocess.run([sys.executable, str(ROOT / job["auditor"]), "--run", str(output), "--reproduce-evaluation"],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    audit = read(audit_path)
    assert audit["status"] == "verified" and audit["exact_original_evaluation_reproduced"]
    assert audit["checkpoint_sha256"] == digest(output / "step_000000000/params.pkl")
    assert audit["trace_sha256"] == digest(output / "step_000000000/preview_trajectories.npz")
    assert digest(ROOT / job["checkpoint"]) == job["checkpoint_sha256"]
    print(json.dumps(dict(status="completed_and_audited", name=name, plan_sha256=digest(OUT / "plan.json"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--names", nargs="+", choices=[f"{g}_s{s}" for s, _ in SOURCES for g in GROUPS])
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare()
    elif args.names:
        for name in args.names:
            execute(name)
    else:
        parser.error("run requires --names")
