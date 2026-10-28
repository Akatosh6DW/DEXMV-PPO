"""Development interventions separating BC trajectory drift and new state inputs."""

import argparse
import copy
import json
from pathlib import Path
import pickle

import jax
import numpy as np

from run_s12_job import digest, enrich_episode, evaluate, initialize_soft, task_digest
from orca_s12_env import OrcaS12Grasp
from s12_soft import SoftConfig, REFERENCE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eval-start", type=int, default=251010)
    p.add_argument("--eval-count", type=int, default=8)
    p.add_argument("--physics-workers", type=int, default=4)
    p.add_argument("--variant", choices=("mask_feedback", "restore_bc_backbone"), default="mask_feedback")
    a = p.parse_args()
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    prior = json.loads((a.run / "run_config.json").read_text())
    if prior["method"] != "guided" or not (a.run / "summary.json").exists():
        raise ValueError("Requires a completed BC+PPO source")
    cfg = SoftConfig(**prior["soft_config"])
    env = OrcaS12Grasp(cfg, "pure", jitter=prior["jitter"])
    if task_digest(env) != prior["task_sha256"] or digest(REFERENCE) != prior["reference_sha256"]:
        raise ValueError("Source task/reference changed")
    net, _ = initialize_soft(env, prior["seed"], prior["std"])
    checkpoint = a.run / "params.pkl"
    params = copy.deepcopy(pickle.loads(checkpoint.read_bytes()))
    layer = params[1]["params"]["hidden_0"]
    original = np.asarray(layer["kernel"])
    if original.shape[0] != 93:
        raise ValueError("Unexpected actor observation layout")
    mask = np.ones(93, bool)
    mask[87:90] = False
    bc_checkpoint = a.run / "bc_params.pkl"
    bc_digest = digest(bc_checkpoint)
    if a.variant == "mask_feedback":
        modified = original.copy()
        modified[mask] = 0.
        layer["kernel"] = modified
        description = "zero nonphase first-layer actor input weights; preserve trained phase weights, biases, later layers"
    else:
        bc = pickle.loads(bc_checkpoint.read_bytes())
        params[1].clear()
        params[1].update(copy.deepcopy(bc[1]))
        layer = params[1]["params"]["hidden_0"]
        modified = np.asarray(layer["kernel"]).copy()
        if np.any(modified[mask] != 0.):
            raise ValueError("BC source was not phase-only")
        modified[mask] = original[mask]
        layer["kernel"] = modified
        description = "restore BC actor backbone, phase weights and biases; keep PPO nonphase input weights"
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "model.xml").write_text(env.native.sim.xml)
    meta = dict(source_run=str(a.run.resolve()), checkpoint_sha256=digest(checkpoint),
        source_seed=prior["seed"], eval_start=a.eval_start, eval_count=a.eval_count,
        ablation=description, variant=a.variant, bc_checkpoint_sha256=bc_digest,
        masked_weight_l2=float(np.linalg.norm(original[mask])),
        soft_config=prior["soft_config"], source_checkpoint_unchanged=True,
        scope="development intervention, not an extra trained policy or held-out comparison")
    (a.output / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
    env.native.enable_workers(a.physics_workers)
    try:
        episodes, traces = evaluate(env, net, params, np.arange(a.eval_start, a.eval_start+a.eval_count))
        episodes = [enrich_episode(e,t,cfg) for e,t in zip(episodes,traces)]
        for episode, trace in zip(episodes,traces):
            np.savez_compressed(a.output / f"trajectory_{episode['seed']}.npz", **trace)
        (a.output / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n")
        summary = dict(status="completed", count=len(episodes),
            success15=float(np.mean([e["success15"] for e in episodes])),
            physical_success15=float(np.mean([e["physical_success15"] for e in episodes])),
            mean_best_hold_sec=float(np.mean([e["best_hold_sec"] for e in episodes])),
            element_collapse_count=sum(e["min_volume_ratio"]<=.05 for e in episodes),
            strain_max=max(e["max_principal_strain"] for e in episodes))
        if digest(checkpoint) != meta["checkpoint_sha256"] or digest(bc_checkpoint) != bc_digest:
            raise ValueError("Source checkpoint changed during diagnosis")
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        env.native.pool.close()


if __name__ == "__main__":
    main()
