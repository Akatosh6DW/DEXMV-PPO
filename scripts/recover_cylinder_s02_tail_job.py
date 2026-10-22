"""Recover the diagnosed post-terminal failure, never suppress active errors."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import traceback

import jax
import numpy as np

from cylinder_s02_terminal_tail import TerminalCylinderGrasp
from orca_cylinder_s02_env import ROOT
import run_orca_suite_job as original

OUT = ROOT/'artifacts/orca_cylinder_s02'
DIAGNOSTIC = OUT/'diagnostics/physics_failure_001'
VALIDATION = OUT/'diagnostics/terminal_tail_validation'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_prefix(trace):
    with np.load(DIAGNOSTIC/'prefix.npz') as prefix:
        length = prefix['alive'].shape[1]
        active = np.concatenate([np.ones((len(prefix['seeds']), 1), bool),
                                 prefix['alive'][:, :-1]], axis=1)
        np.testing.assert_array_equal(trace['alive'][:, :length], prefix['alive'])
        for name in ('qpos', 'qvel'):
            np.testing.assert_array_equal(trace[name][:, :length][active], prefix[name][active])
    return length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    expected = 'artifacts/orca_cylinder_s02/eval/B3_friction/ppo_s1001'
    if job['output'] != expected or job['steps'] or not job['audit']:
        raise ValueError('This recovery is restricted to the diagnosed frozen job')
    validation = json.loads((VALIDATION/'validation.json').read_text())
    diagnosis = json.loads((DIAGNOSTIC/'diagnosis.json').read_text())
    assert validation['passed'] and diagnosis['already_terminated']
    assert diagnosis['first_termination_sec'] < diagnosis['incoming_time']
    guard = ROOT/'scripts/cylinder_s02_terminal_tail.py'
    assert validation['guard_sha256'] == digest(guard)
    record = next(r for r in validation['records'] if r['benchmark'] == 'B3_friction')
    assert record['job_sha256'] == digest(args.job)
    assert record['checkpoint_sha256'] == digest(ROOT/job['checkpoint'])
    assert record['prefix_sha256'] == digest(DIAGNOSTIC/'prefix.npz')
    protocol = json.loads((OUT/'protocol.json').read_text())
    for name, expected_hash in protocol['source_hashes'].items():
        assert digest(ROOT/name) == expected_hash, name
    run = ROOT/job['output']
    amendment = dict(
        scope='Absorbing physics after first done; active physics and scoring unchanged',
        job=job['output'].removeprefix('artifacts/orca_cylinder_s02/'),
        job_sha256=digest(args.job), protocol_sha256=digest(OUT/'protocol.json'),
        diagnosis_sha256=digest(DIAGNOSTIC/'diagnosis.json'),
        validation_sha256=digest(VALIDATION/'validation.json'),
        sources={str(p.relative_to(ROOT)): digest(p) for p in (
            guard, Path(__file__), ROOT/'scripts/audit_cylinder_s02_tail.py')})
    amendment_path = OUT/'terminal_tail_amendment.json'
    with amendment_path.open('x') as stream:
        stream.write(json.dumps(amendment, indent=2)+'\n')

    base_evaluator = original.evaluator

    def checked_evaluator(env, net, seeds):
        evaluate = base_evaluator(env, net, seeds)

        def checked(params):
            initial, trace = jax.device_get(evaluate(params))
            length = check_prefix(trace)
            _, episodes = original.summarize(trace, env, seeds)
            expected_episodes = json.loads((VALIDATION/'B3_friction_ppo_s1001_episodes.json').read_text())
            assert episodes == expected_episodes
            assert not episodes[diagnosis['index'][0]]['clean_success30']
            np.savez_compressed(run/'recovery_prefix.npz',
                                **{k: trace[k][:, :length] for k in ('qpos', 'qvel', 'alive')})
            evidence = dict(passed=True, active_prefix_exact=True, episodes_exact=True,
                            prefix_sha256=digest(run/'recovery_prefix.npz'),
                            amendment_sha256=digest(amendment_path))
            (run/'recovery_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
            return initial, trace

        return checked

    original.MODEL = OUT/'inputs/model.xml'
    original.REFERENCE = OUT/'inputs/reference.npz'
    original.RAW_REFERENCE = OUT/'inputs/unrefined.npz'
    original.DATASET = OUT/'inputs/demos/trajectories.npz'
    original.OrcaSuiteGrasp = TerminalCylinderGrasp
    original.evaluator = checked_evaluator
    sys.argv[1:] = ['--job', str(args.job)]
    log = OUT/'logs'/(args.job.stem+'.log')
    with log.open('x') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        try:
            original.main()
        except Exception:
            traceback.print_exc()
            raise
    config = json.loads((run/'run_config.json').read_text())
    config['evaluation_recovery'] = dict(amendment_sha256=digest(amendment_path),
                                         tail='Absorbing after original first done')
    (run/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    print('RECOVERY_COMPLETE', job['output'], flush=True)


if __name__ == '__main__':
    main()
