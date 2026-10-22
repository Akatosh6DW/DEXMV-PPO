"""Prove terminal-tail recovery preserves active rollouts and original scores."""

import hashlib
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from cylinder_s02_terminal_tail import TerminalCylinderGrasp
from orca_cylinder_s02_env import ROOT
from run_orca_suite_job import evaluator, summarize
from train_orca_matched import initialize_matched

OUT = ROOT/'artifacts/orca_cylinder_s02'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = OUT/'diagnostics/terminal_tail_validation'
    destination.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_pmap_shmap_merge', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    jax.config.update('jax_compilation_cache_dir', str(ROOT/'artifacts/orca30/jax_cache'))
    assert jax.default_backend() == 'gpu'
    records = []
    cases = [('B0_nominal', 'ppo'), ('B0_nominal', 'guided'),
             ('B0_nominal', 'residual'), ('B3_friction', 'ppo')]
    for benchmark, method in cases:
        job_path = OUT/'jobs'/('eval__%s__%s_s1001.json' % (benchmark, method))
        job = json.loads(job_path.read_text())
        arm = 'residual' if method == 'residual' else 'pure'
        env = TerminalCylinderGrasp(OUT/'inputs/model.xml', arm, OUT/'inputs/reference.npz',
                                   benchmark=benchmark, audit=True)
        net, _ = initialize_matched(env, job['seed'], job['std'])
        params = pickle.loads((ROOT/job['checkpoint']).read_bytes())
        # Match the original batch width: smaller GPU GEMMs can round
        # differently and perturb subsequent contact dynamics.
        count = job['eval_count']
        seeds = np.arange(40000, 40000+count)
        initial, trace = jax.device_get(evaluator(env, net, seeds)(params))
        assert all(np.isfinite(v).all() for v in trace.values())
        row, episodes = summarize(trace, env, seeds)
        record = dict(benchmark=benchmark, method=method, episodes=count,
                      job_sha256=digest(job_path), checkpoint_sha256=digest(ROOT/job['checkpoint']))
        if benchmark == 'B3_friction':
            prefix_path = OUT/'diagnostics/physics_failure_001/prefix.npz'
            with np.load(prefix_path) as prefix:
                length = prefix['alive'].shape[1]
                active = np.concatenate([np.ones((count, 1), bool), prefix['alive'][:, :-1]], axis=1)
                np.testing.assert_array_equal(trace['alive'][:, :length], prefix['alive'])
                for name in ('qpos', 'qvel'):
                    np.testing.assert_array_equal(trace[name][:, :length][active], prefix[name][active])
            np.savez_compressed(destination/'B3_friction_ppo_s1001_trace.npz', **trace)
            (destination/'B3_friction_ppo_s1001_episodes.json').write_text(json.dumps(episodes, indent=2)+'\n')
            assert not episodes[75]['success30'] and not episodes[75]['clean_success30']
            record.update(active_prefix_exact=True, prefix_steps=length,
                          prefix_sha256=digest(prefix_path), failed_case_remains_failure=True)
        else:
            run = OUT/'eval'/benchmark/(method+'_s1001')/'step_000000000'
            prior = json.loads((run/'episodes.json').read_text())[:count]
            assert episodes == prior, (method, episodes, prior)
            with np.load(run/'preview_trajectories.npz') as original:
                keep = original['active'].shape[0]
                active = trace['active'][:keep]
                np.testing.assert_array_equal(active, original['active'])
                np.testing.assert_array_equal(trace['alive'][:keep], original['alive'])
                for name in original.files:
                    np.testing.assert_array_equal(trace[name][:keep][active], original[name][active])
            record.update(active_trace_exact=True, episode_statistics_exact=True)
        records.append(record)
        print(json.dumps(record), flush=True)
    result = dict(passed=True, records=records,
                  guard_sha256=digest(ROOT/'scripts/cylinder_s02_terminal_tail.py'),
                  validation_sha256=digest(Path(__file__)))
    (destination/'validation.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
