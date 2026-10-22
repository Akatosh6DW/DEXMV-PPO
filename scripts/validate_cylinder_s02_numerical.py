"""Check that the numerical-failure recorder leaves a valid batch unchanged."""

import json
import pickle
from pathlib import Path

import jax
import numpy as np

from cylinder_s02_tail_v2 import ROOT, OUT, digest
from cylinder_s02_numerical_failure import FailureCylinderGrasp
from run_orca_suite_job import evaluator, summarize
from train_orca_matched import initialize_matched


def main():
    output = OUT/'diagnostics/numerical_validation'
    output.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_pmap_shmap_merge', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    jax.config.update('jax_compilation_cache_dir', str(ROOT/'artifacts/orca30/jax_cache'))
    job_path = OUT/'jobs/eval__B3_friction__guided_s1006.json'
    job = json.loads(job_path.read_text())
    FailureCylinderGrasp.evidence_dir = output/'events'
    env = FailureCylinderGrasp(OUT/'inputs/model.xml', 'pure', OUT/'inputs/reference.npz',
                               benchmark='B3_friction', audit=True)
    net, _ = initialize_matched(env, job['seed'], job['std'])
    params = pickle.loads((ROOT/job['checkpoint']).read_bytes())
    seeds = np.arange(40000, 40200)
    initial, trace = jax.device_get(evaluator(env, net, seeds)(params))
    assert not env.native.events
    _, episodes = summarize(trace, env, seeds)
    prior = ROOT/job['output']/'step_000000000'
    assert episodes == json.loads((prior/'episodes.json').read_text())
    with np.load(prior/'initial_states.npz') as original:
        np.testing.assert_array_equal(initial, original['qpos'])
    with np.load(prior/'preview_trajectories.npz') as original:
        active = trace['active'][:4]
        for name in original.files:
            np.testing.assert_array_equal(trace[name][:4][active], original[name][active])
    result = dict(passed=True, episodes=200, statistics_exact=True, active_previews_exact=True,
                  job_sha256=digest(job_path), checkpoint_sha256=digest(ROOT/job['checkpoint']),
                  helper_sha256=digest(ROOT/'scripts/cylinder_s02_numerical_failure.py'),
                  validation_source_sha256=digest(Path(__file__)))
    (output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
