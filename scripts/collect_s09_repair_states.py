"""Collect learner-visited states labelled by the frozen video-derived teacher.

Teacher mixing is used ONLY for training-data collection, never evaluation.
No successful/failing episodes are removed. Original validation episodes remain
untouched at the end of the dataset, with an explicit training split.
"""

import argparse
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np

import run_s09_job as runner
from retarget_s09_reference import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=111000)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--beta", type=float, default=.5)
    a = p.parse_args()
    if not 0 <= a.beta <= 1 or a.count % 8:
        p.error("Use beta in [0,1], count divisible by8")
    assert jax.default_backend() == "gpu"
    a.output.mkdir(parents=True, exist_ok=False)
    env = runner.OrcaS09Grasp(a.input/"model.xml", "pure", a.input/"reference.npz")
    net, _ = runner.initialize_matched(env, 9101, .3)
    params = pickle.loads(a.checkpoint.read_bytes())
    policy = runner.networks.make_inference_fn(net)(params, deterministic=True)

    def episode(seed):
        key = jax.random.PRNGKey(seed)
        state = env.reset(key)
        def step(carry, _):
            state, alive, key = carry
            key, sub = jax.random.split(key)
            expert = env.ctrl_to_action(env.reference_base(state.info))
            learner = policy(state.obs, sub)[0]
            action = a.beta * expert + (1-a.beta) * learner
            nxt = jax.lax.cond(alive, lambda s: env.step(s, action), lambda s: s, state)
            row = dict(obs=state.obs["state"], action=expert, active=alive)
            return (nxt, alive & (nxt.done == 0), key), row
        (final, alive, _), trace = jax.lax.scan(step, (state, jp.bool_(True), key), None, length=800)
        return trace, final.metrics["success30"] * alive

    collect = jax.jit(jax.vmap(episode))
    batches, success = [], []
    seeds = np.arange(a.start, a.start + a.count, dtype=np.int32)
    for offset in range(0, a.count, 8):
        trace, result = jax.device_get(collect(jp.asarray(seeds[offset:offset+8])))
        batches.append(trace)
        success.extend(np.asarray(result).tolist())
        print(json.dumps(dict(episodes=offset+8, mixed_collection_success=float(np.mean(success)))), flush=True)
    new = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    manifest = json.loads((a.dataset.parent / "dataset_manifest.json").read_text())
    assert manifest["dataset_sha256"] == digest(a.dataset)
    assert manifest["model_sha256"] == digest(a.input/"model.xml")
    assert manifest["reference_sha256"] == digest(a.input/"reference.npz")
    with np.load(a.dataset, allow_pickle=False) as z:
        old = {key: z[key] for key in ("obs", "action", "active")}
    split = manifest.get("train_episodes", len(old["obs"]) * 3 // 4)
    merged = {key: np.concatenate([old[key][:split], new[key], old[key][split:]]) for key in old}
    path = a.output / "trajectories.npz"
    np.savez_compressed(path, **merged)
    manifest.update(dataset_sha256=digest(path), train_episodes=split+a.count,
        parent_dataset=str(a.dataset.resolve()), parent_dataset_sha256=digest(a.dataset),
        teacher_mixing_beta=a.beta, learner_checkpoint=str(a.checkpoint.resolve()),
        learner_checkpoint_sha256=digest(a.checkpoint), new_training_seeds=seeds.tolist(),
        collection_success=success, collection_active_steps=int(new["active"].sum()),
        collection_interpretation="Mixed controller rollouts, NOT standalone BC or PPO test results",
        retained_failures=True, validation_unchanged=True, source_sha256=digest(Path(__file__)))
    (a.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
