"""Measure per-orientation actor-gradient conflict without updating weights.

Fresh stochastic rollouts start from the already used development initials.
This is a mechanism probe, not a held-out performance test or a replay of the
historical optimizer trajectory. Each loss uses the real 32x20 minibatch size.
"""

import argparse
import functools
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np

import run_s09_ppo_improvement as improvement
from brax.training import acting
from brax.training.agents.ppo import losses, networks
from mujoco_playground._src import wrapper
from retarget_s09_reference import digest
from run_s09_bc_conditioned import ModeConditionedGrasp

ROOT = improvement.runner.ROOT
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018"


def actor_terms(policy, value, normalizer, data, key, net):
    data = jax.tree_util.tree_map(lambda x:jp.swapaxes(x, 0, 1), data)
    logits = net.policy_network.apply(normalizer, policy, data.observation)
    baseline = net.value_network.apply(normalizer, value, data.observation)
    terminal = jax.tree_util.tree_map(lambda x:x[-1], data.next_observation)
    bootstrap = net.value_network.apply(normalizer, value, terminal)
    truncation = data.extras["state_extras"]["truncation"]
    _, advantage = losses.compute_gae(truncation, (1-data.discount)*(1-truncation),
        data.reward, baseline, bootstrap, lambda_=.95, discount=.99)
    advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
    dist = net.parametric_action_distribution
    ratio = jp.exp(dist.log_prob(logits, data.extras["policy_extras"]["raw_action"])
                   - data.extras["policy_extras"]["log_prob"])
    objective = -jp.minimum(ratio*advantage, jp.clip(ratio, .7, 1.3)*advantage)
    objective -= .0001*dist.entropy(logits, key)
    modes = jp.argmax(data.observation["state"][..., 95:101], axis=-1)
    # Sum of the six contributions is exactly the actor part of Brax's loss.
    contributions = jp.mean(objective[..., None]*jax.nn.one_hot(modes, 6), axis=(0, 1))
    return contributions, (advantage, modes)


def gradient_measurement(net, params, data, key):
    stats, policy, value = params
    fn = lambda p:actor_terms(p, value, stats, data, key, net)[0]
    grouped = jax.jacrev(fn)(policy)
    matrix = jp.concatenate([x.reshape(6, -1) for x in jax.tree_util.tree_leaves(grouped)], axis=1)
    standard = jax.grad(lambda p: losses.compute_ppo_loss(
        losses.PPONetworkParams(policy=p, value=value), stats, data, key,
        ppo_network=net, discounting=.99, gae_lambda=.95, entropy_cost=.0001,
        clipping_epsilon=.3)[0])(policy)
    standard = jp.concatenate([x.reshape(-1) for x in jax.tree_util.tree_leaves(standard)])
    _, (advantage, modes) = actor_terms(policy, value, stats, data, key, net)
    return matrix, standard, advantage, modes


def summarize_gradients(matrix, counts):
    matrix = np.asarray(matrix, dtype=np.float64)
    counts = np.asarray(counts)
    if matrix.ndim != 2 or matrix.shape[0] != 6 or counts.shape != (6,):
        raise ValueError("Expected six gradient vectors and six counts")
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite gradients")
    norms = np.linalg.norm(matrix, axis=1)
    cosine = [[float(matrix[i]@matrix[j]/(norms[i]*norms[j]))
               if norms[i]>0 and norms[j]>0 and counts[i]>0 and counts[j]>0 else None
               for j in range(6)] for i in range(6)]
    total = matrix.sum(axis=0)
    # Positive means descending the total gradient increases this mode's loss
    # to first order. This is an SGD diagnostic, not the actual Adam update.
    projected = [-float(row@total) for row in matrix]
    return dict(counts=counts.tolist(), norms=norms.tolist(), cosine=cosine,
                full_batch_sgd_first_order_loss_change=projected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--repetitions", type=int, default=2, choices=(1, 2))
    parser.add_argument("--batches", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    if not args.allow_cpu:
        assert jax.default_backend() == "gpu"
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = BASE/"E30_bc_fast_critic_continue_s14112/step_000491520/params.pkl"
    cfg = json.loads((checkpoint.parent.parent/"run_config.json").read_text())
    model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
    assert digest(model) == cfg["model_sha256"]
    env = ModeConditionedGrasp(model, "pure")
    net, _ = improvement.runner.initialize_matched(env, 14122, .03)
    params = pickle.loads(checkpoint.read_bytes())
    policy = networks.make_inference_fn(net)(params, deterministic=False)
    env = wrapper.wrap_for_brax_training(env, episode_length=800, action_repeat=1, full_reset=True)
    collect = jax.jit(functools.partial(acting.generate_unroll, env, policy=policy,
        unroll_length=20, extra_fields=("truncation",)))
    evaluate = jax.jit(functools.partial(gradient_measurement, net))
    rows = []
    for rep in range(args.repetitions):
        seeds = jp.arange(130000+32*rep, 130032+32*rep)
        state = jax.jit(env.reset)(jax.vmap(jax.random.PRNGKey)(seeds))
        key = jax.random.PRNGKey(19101+rep)
        for batch in range(args.batches):
            fragments = []
            for _ in range(4):
                key, subkey = jax.random.split(key)
                state, data = collect(env_state=state, key=subkey)
                fragments.append(data)
            data = jax.tree_util.tree_map(lambda *xs:jp.stack(xs).swapaxes(1, 2).reshape((-1,)+xs[0].swapaxes(0, 1).shape[1:]), *fragments)
            assert data.reward.shape == (128, 20)
            key, perm_key = jax.random.split(key)
            order = jax.random.permutation(perm_key, 128)
            host = jax.device_get(data)
            # Local trusted Transition pickle preserves structure for exact replay.
            path = args.output/f"rep{rep}_batch{batch}_transitions.pkl"
            path.write_bytes(pickle.dumps(host))
            matrices, total_counts = [], np.zeros(6, dtype=int)
            max_error = 0.
            for mini in range(4):
                part = jax.tree_util.tree_map(lambda x:x[order[mini*32:(mini+1)*32]], data)
                key, subkey = jax.random.split(key)
                matrix, standard, advantage, modes = jax.device_get(evaluate(params, part, subkey))
                np.testing.assert_allclose(matrix.sum(axis=0), standard, atol=2e-5, rtol=2e-4)
                error = float(np.max(np.abs(matrix.sum(axis=0)-standard)))
                counts = np.bincount(modes.ravel(), minlength=6)
                row = dict(repetition=rep, rollout_batch=batch, minibatch=mini,
                    gradient_sum_max_error=error,
                    mean_advantage_by_mode=[float(advantage[modes==m].mean()) if counts[m] else None for m in range(6)],
                    **summarize_gradients(matrix, counts))
                rows.append(row)
                matrices.append(matrix)
                total_counts += counts
                max_error = max(max_error, error)
            aggregate = dict(repetition=rep, rollout_batch=batch, minibatch="aggregate_frozen_weights",
                gradient_sum_max_error=max_error, transition_sha256=digest(path),
                **summarize_gradients(np.mean(matrices, axis=0), total_counts))
            rows.append(aggregate)
            print(json.dumps(aggregate), flush=True)
    report = dict(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest(checkpoint),
        backend=jax.default_backend(), optimizer_updates=0, collection_seeds=[19101+i for i in range(args.repetitions)],
        scope="Frozen-weight stochastic on-policy probe on consumed development initials; resets remain enabled. Not historical training minibatches, not a success evaluation, and not evidence of causality by itself.",
        comparison="Per-mode contributions to the original PPO actor loss, including entropy; same per-minibatch advantage normalization. Cosines and projected SGD changes are not Adam updates.",
        source_sha256=digest(Path(__file__)), installed_brax_loss_sha256=digest(Path(losses.__file__)),
        installed_brax_acting_sha256=digest(Path(acting.__file__)), rows=rows)
    (args.output/"gradient_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(status="completed", measurements=len(rows), optimizer_updates=0)))


if __name__ == "__main__":
    main()
