"""Measure inherited value calibration on the actual zero-residual policy.

This is stochastic on-policy data collection, not PPO or a success benchmark.
Finite-episode Monte Carlo targets have sampling variance; a large error alone
does not establish that critic error caused later policy degradation.
"""

import argparse
import json
from pathlib import Path
import pickle
import shutil

import jax
import jax.numpy as jp
import numpy as np

import run_s09_learned_base_residual as learned
from run_s09_learned_residual_experts import residual_expert_factory
from pretrain_s09_critic import collector, returns_to_go
from retarget_s09_reference import digest
from analyze_s09_bc_confirmation import write_csv


def calibration(prediction, target, mask):
    prediction, target, mask = np.asarray(prediction), np.asarray(target), np.asarray(mask, bool)
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Value, target and mask shapes must match")
    if not np.all(np.isfinite(prediction[mask])) or not np.all(np.isfinite(target[mask])):
        raise ValueError("Nonfinite calibration data")
    if not mask.any():
        return dict(n=0, rmse=None, bias=None, mean_prediction=None, mean_return=None)
    error = prediction[mask].astype(float)-target[mask]
    return dict(n=int(mask.sum()), rmse=float(np.sqrt(np.mean(error**2))),
                bias=float(np.mean(error)), mean_prediction=float(prediction[mask].mean()),
                mean_return=float(target[mask].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args()
    if args.episodes not in (8, 32):
        parser.error("Use the predeclared 8-case smoke or 32-case diagnostic")
    allowed = learned.runner.ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"
    if not args.output.resolve().is_relative_to(allowed):
        parser.error("Keep diagnostic output separate from original experiments")
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_pmap_shmap_merge", False)
    cfg = json.loads((args.run/"run_config.json").read_text())
    info = json.loads((args.run/"learned_base_config.json").read_text())
    if cfg["checkpoint"] or cfg["method"] != "residual":
        parser.error("Use a fresh residual run's zero-step checkpoint")
    for name in ("run_s09_learned_base_residual.py", "run_s09_learned_residual_experts.py"):
        assert digest(Path(__file__).parent/name) == cfg["sources"][name]
    checkpoint = args.run/"step_000000000/params.pkl"
    params = jax.tree_util.tree_map(jp.asarray, pickle.loads(checkpoint.read_bytes()))
    assert len(params[1]["experts"]) == 6
    for actor in params[1]["experts"]:
        np.testing.assert_array_equal(actor["params"]["hidden_2"]["kernel"], 0)
        np.testing.assert_array_equal(actor["params"]["hidden_2"]["bias"][:17], 0)
    model = Path(cfg["input"])/"model.xml"
    base = Path(info["base_checkpoint"])
    assert digest(base) == info["base_sha256"]
    action, _, _ = learned.load_base(base, model)
    env = learned.LearnedBaseGrasp(model, base_action=action, residual_bound=info["residual_bound_rad"])
    net = residual_expert_factory(action)(dict(state=(123,), privileged_state=(140,)), 17)
    collect = collector(env, net, deterministic=False)
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = np.arange(240000, 240000+args.episodes)
    protocol = dict(status="running", seeds=seeds.tolist(), checkpoint_sha256=digest(checkpoint),
        base_sha256=digest(base), model_sha256=digest(model), source_run=str(args.run.resolve()),
        source_sha256=digest(Path(__file__)), discount=.99, stochastic_policy=True,
        residual_std=cfg["std"], residual_bound_rad=info["residual_bound_rad"],
        policy_updates=0, critic_updates=0, failures_retained=True,
        target="Finite 40s discounted R0 return, zero beyond episode; not the PPO bootstrap target",
        scope="New diagnostic/training states, not held-out success evaluation; no native replay audit claimed")
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    batches = []
    for start in range(0, len(seeds), 8):
        batches.append(jax.device_get(collect(params, jp.asarray(seeds[start:start+8], jp.int32))))
        print(json.dumps(dict(collected=start+8)), flush=True)
    initial, trace = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *batches)
    target = returns_to_go(trace["reward"], trace["active"])
    observations = jp.asarray(trace["privileged_obs"])
    predict = jax.jit(lambda x: net.value_network.apply(params[0], params[2],
        dict(state=x[...,:123], privileged_state=x)))
    prediction = np.asarray(predict(observations))
    mode = np.argmax(trace["privileged_obs"][:,0,95:101], axis=-1)
    rows = []
    frames = np.arange(800)[None,:]
    # Excluding the final 10s separates late finite-horizon effects from earlier errors.
    for group in range(-1, 6):
        cases = np.ones(len(seeds), bool) if group == -1 else mode == group
        for phase, times in (("all", frames>=0), ("acquisition", frames<60),
                             ("released_before_30s", (frames>=60)&(frames<600))):
            mask = trace["active"] & cases[:,None] & times
            rows.append(dict(mode=group, phase=phase, episodes=int(cases.sum()),
                             **calibration(prediction, target, mask)))
    data = args.output/"critic_calibration.npz"
    np.savez_compressed(data, **trace, initial_qpos=initial, seeds=seeds,
                        returns=target, value_prediction=prediction, mode=mode)
    write_csv(args.output/"calibration.csv", rows)
    summary = dict(protocol, status="completed", rows=rows,
        actual_environment_interactions=int(trace["active"].sum()), data_sha256=digest(data),
        caveat="Correlated frames and single sampled return per initial state; not a causal critic ablation or uncertainty interval")
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    shutil.copyfile(Path(__file__), args.output/Path(__file__).name)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
