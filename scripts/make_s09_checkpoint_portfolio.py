"""Select historical full-target actors by pose using development data only.

This is checkpoint selection/assembly, not new PPO training. The three-source
pool and tie rule are fixed; heldout outcomes never participate in selection.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np

from confirm_s09_late_experts import ROOT, BASE
from retarget_s09_reference import digest
from run_s09_mode_experts import ARCHITECTURE, BASE_FACTORY, network_factory

OUT = BASE/"confirmation_portfolio_20261019"


def choose_sources(counts):
    values = np.asarray(counts)
    if values.shape != (3,6) or not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
        raise ValueError("Require three sources and six nonnegative integer success counts")
    # Sources are ordered BC, stage, endpoint: ties preserve the earlier source.
    return np.argmax(values, axis=0).tolist()


def main():
    jax.config.update("jax_default_matmul_precision", "highest")
    read = lambda p: json.loads(p.read_text())
    folders = [BASE/"E27_bc_fast_critic_s14102/step_000000000",
        BASE/"E30_bc_fast_critic_continue_s14112/step_000491520",
        BASE/"E78_bc_late_experts_long/step_000655360"]
    names = ["bc", "stage", "expert"]
    configs = [read(p.parent/"run_config.json") for p in folders]
    counts, parameters, provenance = [], [], []
    initial = None
    for name, folder, cfg in zip(names, folders, configs):
        assert cfg["eval_start"] == 130000 and cfg["eval_count"] == 64
        for key in ("model_sha256", "reference_sha256", "env_config"):
            assert cfg[key] == configs[0][key]
        with np.load(folder/"initial_states.npz") as z:
            np.testing.assert_array_equal(z["seeds"], np.arange(130000,130064))
            if initial is None:
                initial = z["qpos"].copy()
            np.testing.assert_array_equal(z["qpos"], initial)
        episodes = read(folder/"episodes.json")
        counts.append([sum(r["success30"] for r in episodes if r["selected_reference"] == m) for m in range(6)])
        parameters.append(pickle.loads((folder/"params.pkl").read_bytes()))
        provenance.append(dict(name=name, checkpoint=str((folder/"params.pkl").relative_to(ROOT)),
            checkpoint_sha256=digest(folder/"params.pkl"), episodes_sha256=digest(folder/"episodes.json")))
    choices = choose_sources(counts)
    actors = tuple(parameters[k][1]["experts"][m] if k == 2 else parameters[k][1]
                   for m,k in enumerate(choices))
    # Frozen S09 uses raw observations (normalize_observations=False). Running
    # statistics still accumulate in PPO; keep endpoint stats with its critic,
    # and check each source with its own statistics rather than assuming equal.
    combined = (parameters[2][0], {"experts":actors}, parameters[2][2])
    base = BASE_FACTORY(dict(state=(123,), privileged_state=(140,)), 17)
    net = network_factory(dict(state=(123,), privileged_state=(140,)), 17)
    errors = []
    for m,k in enumerate(choices):
        with np.load(folders[k]/"preview_trajectories.npz") as z:
            obs = z["obs"][z["active"]].copy()
        obs[:,95:101] = 0
        obs[:,95+m] = 1
        inputs = dict(state=jp.asarray(obs))
        a = np.asarray(base.parametric_action_distribution.mode(base.policy_network.apply(parameters[k][0], actors[m], inputs)))
        b = np.asarray(net.parametric_action_distribution.mode(net.policy_network.apply(combined[0], combined[1], inputs)))
        np.testing.assert_allclose(a,b,atol=1e-5,rtol=0)
        errors.append(float(np.max(np.abs(a-b))))
    OUT.mkdir(exist_ok=False)
    candidate = OUT/"candidate"
    candidate.mkdir()
    checkpoint = candidate/"params.pkl"
    checkpoint.write_bytes(pickle.dumps(jax.device_get(combined)))
    cfg = dict(configs[2], policy_architecture=ARCHITECTURE,
        policy_selection="per-pose historical-checkpoint selection on 64 development states", extra_training_steps=0)
    (candidate/"run_config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    selection = dict(status="assembled_not_newly_trained", sources=provenance,
        development_seeds=[130000,130063], source_success_counts=counts,
        selected_source_by_mode=[names[k] for k in choices], tie_rule="earliest source BC then stage then expert",
        predicted_development_successes=sum(counts[k][m] for m,k in enumerate(choices)),
        predicted_not_yet_executed=True, max_same_input_action_errors=errors,
        extra_ppo_steps=0, inherited_path_online_steps=1310720,
        normalization="raw observations; source running statistics differ but are not used by the frozen policy factory",
        actor_parameter_multiplier=6, candidate_sha256=digest(checkpoint), source_sha256=digest(Path(__file__)))
    (OUT/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    jobs = []
    for name, path, multiplier, budget in (
        ("bc", folders[0]/"params.pkl", 1, 0),
        ("expert", folders[2]/"params.pkl", 6, 1310720),
        ("portfolio", checkpoint, 6, 1310720)):
        jobs.append(dict(name=name, checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=digest(path),
            actor_parameter_multiplier=multiplier, path_online_steps=budget, training_seed=14122,
            output=str((OUT/"eval"/name).relative_to(ROOT)),
            entry="scripts/evaluate_s09_failure_quality.py" if multiplier==1 else "scripts/evaluate_s09_mode_expert_quality.py",
            auditor="scripts/audit_s09_bc_repair.py" if multiplier==1 else "scripts/audit_s09_mode_experts.py"))
    old_plan = read(BASE/"confirmation_experts_20261019/plan.json")
    for path, expected in old_plan["execution_sources"].items():
        assert digest(ROOT/path) == expected
    sources = dict(old_plan["execution_sources"])
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__))
    sources["scripts/run_s09_portfolio_confirmation.py"] = digest(ROOT/"scripts/run_s09_portfolio_confirmation.py")
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input=old_plan["input"], eval_start=180000, eval_count=256, jobs=jobs,
        model_sha256=cfg["model_sha256"], reference_sha256=cfg["reference_sha256"], env_config=cfg["env_config"],
        execution_sources=sources, selection_sha256=digest(OUT/"selection.json"),
        comparisons=["portfolio minus bc", "portfolio minus expert"],
        statistics="Wilson rates; paired bootstrap 20000 draws seed20261015; exact McNemar with Holm across two contrasts",
        limitations="New method designed after 170000 confirmation; 180000 range not used for source selection. One ancestral training path, six actors, zero additional PPO updates, extra development selection cost.")
    (OUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    print(json.dumps(dict(selection=selection, plan_sha256=digest(OUT/"plan.json"))))


if __name__ == "__main__":
    main()
