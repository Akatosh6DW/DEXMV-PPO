"""Run a sealed remaining evaluation with absorbing post-terminal physics."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from cylinder_s02_tail_v2 import ROOT, OUT, MANIFEST, digest, verify
from cylinder_s02_terminal_tail import TerminalCylinderGrasp
import run_orca_suite_job as original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    seal = verify()
    record = next(r for r in seal['jobs'] if (ROOT/r['path']).resolve() == args.job.resolve())
    assert digest(args.job) == record['sha256']
    job = json.loads(args.job.read_text())
    assert job['steps'] == 0 and job['audit']
    original.MODEL = OUT/'inputs/model.xml'
    original.REFERENCE = OUT/'inputs/reference.npz'
    original.RAW_REFERENCE = OUT/'inputs/unrefined.npz'
    original.DATASET = OUT/'inputs/demos/trajectories.npz'
    original.OrcaSuiteGrasp = TerminalCylinderGrasp
    run = ROOT/job['output']
    if record['name'] == 'eval/B3_friction/ppo_s1002':
        base_evaluator = original.evaluator

        def checked_evaluator(env, net, seeds):
            evaluate = base_evaluator(env, net, seeds)

            def checked(params):
                initial, trace = jax.device_get(evaluate(params))
                diagnostic = OUT/'diagnostics/physics_failure_002'
                with np.load(diagnostic/'prefix.npz') as old:
                    length = old['alive'].shape[1]
                    active = np.concatenate([np.ones((200, 1), bool), old['alive'][:, :-1]], axis=1)
                    np.testing.assert_array_equal(trace['alive'][:, :length], old['alive'])
                    for name in ('qpos', 'qvel'):
                        np.testing.assert_array_equal(trace[name][:, :length][active], old[name][active])
                np.savez_compressed(run/'recovery_prefix.npz',
                                    **{k: trace[k][:, :length] for k in ('qpos', 'qvel', 'alive')})
                index = json.loads((diagnostic/'diagnosis.json').read_text())['index'][0]
                assert not trace['alive'][index, -1]
                return initial, trace

            return checked

        original.evaluator = checked_evaluator
    original.main()
    config = json.loads((run/'run_config.json').read_text())
    config['terminal_tail_v2_sha256'] = digest(MANIFEST)
    (run/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')


if __name__ == '__main__':
    main()
