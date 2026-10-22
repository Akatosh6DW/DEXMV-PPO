"""Measure policy forgetting on fixed BC validation states, without retraining."""

import argparse
import csv
import inspect
import json
from pathlib import Path
import pickle
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training import distribution
from brax.training.agents.ppo import train, losses

from train_orca_comparison import network_factory
from prepare_cylinder_s02_source import digest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / "run_config.json").read_text())
    if config["method"] != "guided":
        parser.error("This diagnostic is for BC+PPO checkpoints")
    with np.load(config["dataset"]) as z:
        split = len(z["obs"])*3//4
        mask = z["active"].astype(bool)
        train_obs = z["obs"][:split][mask[:split]]
        obs = z["obs"][split:][mask[split:]]
        target = z["action"][split:][mask[split:]]
    net = network_factory(dict(state=(123,), privileged_state=(140,)), 17)
    model = mujoco.MjModel.from_xml_path(config["model"])
    low, high = model.actuator_ctrlrange.T
    half = (high-low)/2
    policy_apply = jax.jit(lambda params, x: net.policy_network.apply(params[0], params[1], {"state": x}))
    initial = pickle.loads((args.run / "step_000000000/params.pkl").read_bytes())
    initial_logits = np.asarray(policy_apply(initial, jp.asarray(obs)))
    initial_mean = np.tanh(initial_logits[:, :17])
    global_key = jax.random.split(jax.random.PRNGKey(config["seed"]))[0]
    value_key = jax.random.split(global_key)[1]
    random_value = net.value_network.init(value_key)
    value_error = max(float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
                      for x, y in zip(jax.tree_util.tree_leaves(initial[2]), jax.tree_util.tree_leaves(random_value)))
    records, joints = [], []
    train_low, train_high = train_obs.min(0), train_obs.max(0)
    old_dist = net.parametric_action_distribution.create_dist(jp.asarray(initial_logits))
    for step in args.steps:
        path = args.run / f"step_{step:09d}"
        params = pickle.loads((path / "params.pkl").read_bytes())
        logits = np.asarray(policy_apply(params, jp.asarray(obs)))
        mean = np.tanh(logits[:, :17])
        dist = net.parametric_action_distribution.create_dist(jp.asarray(logits))
        kl = np.asarray(dist.kl_divergence(old_dist))
        target_error = (mean-target)*half
        drift = (mean-initial_mean)*half
        # Differential scaling is a local approximation, not a rollout test.
        scale = np.asarray(dist.scale)
        physical_std = half*(1-mean**2)*scale
        row = dict(step=step, checkpoint_sha256=digest(path / "params.pkl"),
                   bc_validation_latent_mse=float(np.mean((logits[:, :17]-np.arctanh(np.clip(target, -.995, .995)))**2)),
                   target_rms_rad=float(np.sqrt(np.mean(target_error**2))),
                   bc_mean_drift_rms_rad=float(np.sqrt(np.mean(drift**2))),
                   bc_mean_drift_max_rad=float(np.max(np.abs(drift))),
                   fixed_state_kl_mean=float(kl.mean()), fixed_state_kl_median=float(np.median(kl)),
                   fixed_state_kl_p95=float(np.quantile(kl, .95)), fixed_state_kl_max=float(kl.max()),
                   latent_std_mean=float(scale.mean()), physical_std_approx_mean_rad=float(physical_std.mean()))
        with np.load(path / "preview_trajectories.npz") as z:
            active = z["active"].astype(bool)
            visited = z["obs"][active]
            outside = (visited < train_low-1e-5) | (visited > train_high+1e-5)
            row["dev_outside_training_box_feature_fraction"] = float(outside.mean())
            row["dev_any_feature_outside_fraction"] = float(np.any(outside, axis=1).mean())
            row["dev_saved_episodes"] = len(z["obs"])
        for j in range(17):
            joints.append(dict(step=step, actuator=model.actuator(j).name,
                target_rms_rad=float(np.sqrt(np.mean(target_error[:, j]**2))),
                bc_mean_drift_rms_rad=float(np.sqrt(np.mean(drift[:, j]**2))),
                physical_std_approx_mean_rad=float(physical_std[:, j].mean())))
        records.append(row)
    for name, rows in (("checkpoint_drift.csv", records), ("joint_drift.csv", joints)):
        with (args.output / name).open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    dependency = {}
    for module in (train, losses, distribution):
        path = Path(inspect.getfile(module))
        name = module.__name__.replace(".", "_")+".py"
        shutil.copy2(path, args.output / name)
        dependency[str(path)] = digest(path)
    defaults = inspect.signature(train.train).parameters
    meta = dict(run=str(args.run.resolve()), validation_observations=len(obs), training_observations=len(train_obs),
        dataset_sha256=digest(config["dataset"]), critic_bc_equals_random_init_max_error=value_error,
        dependency_sha256=dependency,
        ppo_defaults={k: str(defaults[k].default) for k in ("clipping_epsilon", "max_grad_norm", "gae_lambda",
                                                         "desired_kl", "learning_rate_schedule")},
        caveat="Read-only diagnostic, not causal ablation. KL on fixed demonstration-validation states differs from on-policy logged KL. Min/max feature-box coverage is not a density or OOD probability. Physical std uses a local derivative; no simulator rollout performed.")
    (args.output / "manifest.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(dict(metadata=meta, rows=records)), flush=True)


if __name__ == "__main__":
    main()
