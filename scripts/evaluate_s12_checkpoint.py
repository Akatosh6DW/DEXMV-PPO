"""Explicit held-out or transfer evaluation; never silently relax checkpoint binding."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle

from run_s12_job import digest, enrich_episode, evaluate, initialize_soft, task_digest
from orca_s12_env import OrcaS12Grasp
from s12_soft import SoftConfig, REFERENCE
import jax
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--reference", action="store_true")
    p.add_argument("--condition", choices=("nominal", "position5mm", "soft5kpa", "stiff20kpa", "fine"), default="nominal")
    p.add_argument("--eval-start", type=int, required=True)
    p.add_argument("--eval-count", type=int, required=True)
    p.add_argument("--physics-workers", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    if a.reference and a.checkpoint:
        p.error("Reference evaluation has no learned checkpoint")
    prior = json.loads((a.run/"run_config.json").read_text())
    if not (a.run/"summary.json").exists():
        raise ValueError("Source run not completed")
    cfg = SoftConfig(**prior["soft_config"])
    arm = "residual" if prior["method"]=="residual" and not a.reference else "pure"
    original = OrcaS12Grasp(cfg, arm, jitter=prior["jitter"])
    import hashlib
    if hashlib.sha256(original.native.sim.xml.encode()).hexdigest()!=prior["model_sha256"]:
        raise ValueError("Original model differs from trained checkpoint")
    if digest(REFERENCE)!=prior["reference_sha256"]:
        raise ValueError("Original reference changed")
    if task_digest(original)!=prior["task_sha256"]:
        raise ValueError("Original task timing/reward/action contract changed")
    if a.condition=="soft5kpa":
        cfg = replace(cfg, young=5000.)
    elif a.condition=="stiff20kpa":
        cfg = replace(cfg, young=20000.)
    elif a.condition=="fine":
        cfg = replace(cfg, sectors=24, timestep=.0001)
    jitter = .005 if a.condition=="position5mm" else prior["jitter"]
    env = OrcaS12Grasp(cfg, arm, jitter=jitter)
    net, initial = initialize_soft(env, prior["seed"], prior["std"])
    checkpoint = a.checkpoint or a.run/"params.pkl"
    params = initial if a.reference else pickle.loads(checkpoint.read_bytes())
    obs = env.reset(jax.random.PRNGKey(0)).obs
    if (obs["state"].size, obs["privileged_state"].size)!=(prior["actor_dim"], prior["critic_dim"]):
        raise ValueError("Policy observation contract changed")
    np.testing.assert_array_equal(original.native.model.actuator_ctrlrange, env.native.model.actuator_ctrlrange)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/"model.xml").write_text(env.native.sim.xml)
    meta = dict(source_run=str(a.run.resolve()), source_config_sha256=digest(a.run/"run_config.json"),
        checkpoint=None if a.reference else str(checkpoint.resolve()),
        checkpoint_sha256=None if a.reference else digest(checkpoint),
        method="reference" if a.reference else prior["method"], condition=a.condition,
        soft_config=asdict(cfg), jitter=jitter, model_sha256=digest(a.output/"model.xml"),
        reference_sha256=digest(REFERENCE), eval_start=a.eval_start, eval_count=a.eval_count,
        changes=dict(young=cfg.young, sectors=cfg.sectors, timestep=cfg.timestep, jitter=jitter),
        physics_workers=a.physics_workers)
    (a.output/"evaluation_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    if a.physics_workers:
        env.native.enable_workers(a.physics_workers)
    try:
        episodes, traces = evaluate(env, net, params,
            np.arange(a.eval_start, a.eval_start+a.eval_count), reference=a.reference)
        episodes = [enrich_episode(e,t,cfg) for e,t in zip(episodes,traces)]
        for e,trace in zip(episodes,traces):
            np.savez_compressed(a.output/f"trajectory_{e['seed']}.npz", **trace)
        (a.output/"episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        summary = dict(status="completed", count=len(episodes),
            success15=float(np.mean([e["success15"] for e in episodes])),
            physical_success15=float(np.mean([e["physical_success15"] for e in episodes])),
            hold_sec=float(np.mean([e["best_hold_sec"] for e in episodes])),
            strain_max=max(e["max_principal_strain"] for e in episodes),
            depth_max=max(e["max_penetration_m"] for e in episodes))
        (a.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print("DONE", json.dumps(summary), flush=True)
    finally:
        if env.native.pool is not None:
            env.native.pool.close()


if __name__ == "__main__":
    main()
