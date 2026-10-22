"""Recover a diagnosed active numerical failure with explicit censored outcomes."""

import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback

import jax
import numpy as np

from cylinder_s02_tail_v2 import ROOT, OUT, MANIFEST, digest, verify
from cylinder_s02_numerical_failure import FailureCylinderGrasp
import run_cylinder_s02_terminal_job as worker
import run_orca_suite_job as original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    args = parser.parse_args()
    seal = verify()
    record = next(r for r in seal['jobs'] if (ROOT/r['path']).resolve() == args.job.resolve())
    assert digest(args.job) == record['sha256']
    diagnosis = json.loads((args.diagnostic/'diagnosis.json').read_text())
    assert not diagnosis['already_terminated']
    validation_path = OUT/'diagnostics/numerical_validation/validation.json'
    validation = json.loads(validation_path.read_text())
    assert validation['passed']
    assert validation['helper_sha256'] == digest(ROOT/'scripts/cylinder_s02_numerical_failure.py')
    job = json.loads(args.job.read_text())
    assert not job['steps'] and job['audit'] and job['eval_count'] == 200
    run = ROOT/job['output']
    directory = OUT/'numerical_recovery_plans'
    directory.mkdir(exist_ok=True)
    plan_path = directory/(args.job.stem+'.json')
    sources = [Path(__file__), ROOT/'scripts/cylinder_s02_numerical_failure.py',
               ROOT/'scripts/audit_cylinder_s02_numerical.py']
    plan = dict(job=record['name'], job_sha256=record['sha256'],
                policy='Keep all seeds; numerical invalidity is a conservative failure with sensitivity bounds',
                diagnostic=str(args.diagnostic.resolve().relative_to(ROOT)),
                diagnosis_sha256=digest(args.diagnostic/'diagnosis.json'),
                prefix_sha256=digest(args.diagnostic/'prefix.npz'),
                validation_sha256=digest(validation_path),
                v2_sha256=digest(MANIFEST),
                sources={str(p.relative_to(ROOT)): digest(p) for p in sources})
    with plan_path.open('x') as stream:
        stream.write(json.dumps(plan, indent=2)+'\n')
    FailureCylinderGrasp.evidence_dir = run/'physics_failure_states'
    worker.TerminalCylinderGrasp = FailureCylinderGrasp
    base_evaluator = original.evaluator

    def recorded_evaluator(env, net, seeds):
        evaluate = base_evaluator(env, net, seeds)

        def recorded(params):
            initial, trace = jax.device_get(evaluate(params))
            events = env.native.events
            if not events:
                raise RuntimeError('Expected active numerical error did not reproduce')
            invalid = np.zeros_like(trace['alive'], dtype=bool)
            for event in events:
                index = event['batch_index'][0]
                step = int(round(event['incoming_time']/env.dt))
                assert 0 <= step < 800 and trace['active'][index, step]
                event.update(seed=int(seeds[index]), step_index=step,
                             state_sha256=digest(run/'physics_failure_states'/event['state_file']))
                invalid[index, step:] = True
            trace['physics_invalid'] = invalid
            trace['active'] = trace['active'] & ~invalid
            trace['reward'] = np.where(invalid, 0, trace['reward'])
            assert not np.any(trace['alive'] & invalid)
            with np.load(args.diagnostic/'prefix.npz') as old:
                length = old['alive'].shape[1]
                active = np.concatenate([np.ones((200, 1), bool), old['alive'][:, :-1]], axis=1)
                np.testing.assert_array_equal(trace['alive'][:, :length], old['alive'])
                for name in ('qpos', 'qvel'):
                    np.testing.assert_array_equal(trace[name][:, :length][active], old[name][active])
            np.savez_compressed(run/'numerical_prefix.npz',
                                **{k: trace[k][:, :length] for k in ('qpos', 'qvel', 'alive')})
            _, episodes = original.summarize(trace, env, seeds)
            for index in np.flatnonzero(invalid.any(axis=1)):
                assert not episodes[index]['success30'] and not episodes[index]['clean_success30']
            evidence = dict(events=events, invalid_seeds=[int(seeds[i]) for i in np.flatnonzero(invalid.any(axis=1))],
                            plan_sha256=digest(plan_path), active_prefix_exact=True,
                            metric_note='Invalid step and tail excluded; force/hold metrics are censored prefixes')
            (run/'numerical_failures.json').write_text(json.dumps(evidence, indent=2)+'\n')
            return initial, trace

        return recorded

    original.evaluator = recorded_evaluator
    sys.argv[1:] = ['--job', str(args.job)]
    with (OUT/'logs'/(args.job.stem+'.log')).open('x') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        try:
            worker.main()
        except Exception:
            traceback.print_exc()
            raise
    config = json.loads((run/'run_config.json').read_text())
    config['numerical_recovery_plan_sha256'] = digest(plan_path)
    (run/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    print('NUMERICAL_RECOVERY_COMPLETE', record['name'], flush=True)


if __name__ == '__main__':
    main()
