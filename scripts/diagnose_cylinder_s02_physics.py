"""Locate a native-physics failure without changing the frozen experiment."""

import argparse
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks

from orca_cylinder_s02_env import OrcaCylinderS02Grasp, ROOT
from run_orca_suite_job import evaluator
from train_orca_matched import initialize_matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    job = json.loads(args.job.read_text())
    jax.config.update('jax_pmap_shmap_merge', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    jax.config.update('jax_compilation_cache_dir', str(ROOT/'artifacts/orca30/jax_cache'))
    assert jax.default_backend() == 'gpu'
    inputs = ROOT/'artifacts/orca_cylinder_s02/inputs'
    arm = 'pure' if job['method'] in ('ppo', 'guided', 'bc') else 'residual'
    env = OrcaCylinderS02Grasp(inputs/'model.xml', arm, inputs/'reference.npz',
                              benchmark=job['benchmark'], audit=True,
                              variant=job.get('variant', 'video'),
                              raw_reference=inputs/'unrefined.npz')
    net, _ = initialize_matched(env, job['seed'], job.get('std', .3))
    params = pickle.loads((ROOT/job['checkpoint']).read_bytes())
    seeds = np.arange(job['eval_start'], job['eval_start']+job['eval_count'])
    native = env.native
    failure = {}

    def instrumented_batch(mode, *arrays):
        arrays = tuple(np.asarray(x) for x in arrays)
        shape = arrays[0].shape[:-1]
        if not shape:
            return native.single(mode, *arrays)
        values = []
        for index in np.ndindex(shape):
            arguments = tuple(a[index] for a in arrays)
            try:
                values.append(native.single(mode, *arguments))
            except FloatingPointError as error:
                data = native.restore(arguments[0])
                failure.update(index=list(index), seed=int(seeds[index[0]]),
                               incoming_time=float(data.time), mode=mode, error=str(error))
                np.savez_compressed(args.output/'failure_inputs.npz',
                                    **{'argument_%d' % i: a for i, a in enumerate(arguments)},
                                    qpos=data.qpos.copy(), qvel=data.qvel.copy())
                (args.output/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
                raise
        return jax.tree_util.tree_map(
            lambda *items: np.stack(items).reshape((*shape, *items[0].shape)), *values)

    native.batched = instrumented_batch
    try:
        jax.device_get(evaluator(env, net, seeds)(params))
    except Exception:
        if not failure:
            raise
    else:
        raise RuntimeError('The original failure did not reproduce')

    # Recreate the exact policy RNG/state prefix, stopping before the bad step.
    length = int(round(failure['incoming_time']/env.dt))
    policy = networks.make_inference_fn(net)(params, deterministic=True)

    def prefix(key):
        state = env.reset(key)

        def step(carry, _):
            state, alive, key = carry
            key, subkey = jax.random.split(key)
            action = policy(state.obs, subkey)[0]
            nxt = env.step(state, action)
            next_alive = alive & (nxt.done == 0)
            return (nxt, next_alive, key), dict(alive=next_alive, done=nxt.done,
                                               hold=nxt.metrics['hold_sec'],
                                               qpos=nxt.data.qpos, qvel=nxt.data.qvel)

        return jax.lax.scan(step, (state, jp.bool_(True), jax.random.fold_in(key, 987654)),
                            None, length=length)[1]

    traces = jax.device_get(jax.jit(jax.vmap(prefix))(
        jax.vmap(jax.random.PRNGKey)(jp.asarray(seeds, jp.int32))))
    index = failure['index'][0]
    dead = np.flatnonzero(~traces['alive'][index])
    failure.update(prefix_steps=length, already_terminated=bool(len(dead)),
                   first_termination_step=int(dead[0]+1) if len(dead) else None,
                   first_termination_sec=float((dead[0]+1)*env.dt) if len(dead) else None,
                   prefix_max_hold=float(np.max(traces['hold'][index])))
    np.savez_compressed(args.output/'prefix.npz', **traces, seeds=seeds)
    (args.output/'diagnosis.json').write_text(json.dumps(failure, indent=2)+'\n')
    print(json.dumps(failure), flush=True)


if __name__ == '__main__':
    main()
