"""Seal, run, and audit evaluation-only terminal protection for remaining jobs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/orca_cylinder_s02'
MANIFEST = OUT/'terminal_tail_v2.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    seal = json.loads(MANIFEST.read_text())
    for name, expected in seal['hashes'].items():
        assert digest(ROOT/name) == expected, name
    assert seal['protocol_sha256'] == digest(OUT/'protocol.json')
    return seal


def prepare():
    diagnosis = OUT/'diagnostics/physics_failure_002/diagnosis.json'
    assert json.loads(diagnosis.read_text())['already_terminated']
    validation_path = OUT/'diagnostics/terminal_tail_validation/validation.json'
    validation = json.loads(validation_path.read_text())
    assert validation['passed']
    assert validation['guard_sha256'] == digest(ROOT/'scripts/cylinder_s02_terminal_tail.py')
    protocol = json.loads((OUT/'protocol.json').read_text())
    jobs = [r for r in protocol['jobs'] if r['name'].startswith('eval/')
            and not (OUT/r['name']/'summary.json').exists()]
    paths = [Path(__file__), ROOT/'scripts/run_cylinder_s02_terminal_job.py',
             ROOT/'scripts/cylinder_s02_terminal_tail.py', validation_path,
             diagnosis, diagnosis.parent/'prefix.npz']
    seal = dict(scope='Remaining formal evaluations only; original active physics and scoring',
                protocol_sha256=digest(OUT/'protocol.json'), jobs=jobs,
                hashes={str(p.relative_to(ROOT)): digest(p) for p in paths})
    with MANIFEST.open('x') as stream:
        stream.write(json.dumps(seal, indent=2)+'\n')
    print(json.dumps(dict(sealed=True, remaining_jobs=len(jobs))), flush=True)


def audit(allow_partial):
    seal = verify()
    checked, missing, tails = [], [], 0
    for record in seal['jobs']:
        run = OUT/record['name']
        if not (run/'summary.json').exists():
            missing.append(record['name'])
            continue
        config = json.loads((run/'run_config.json').read_text())
        assert config['terminal_tail_v2_sha256'] == digest(MANIFEST)
        assert config['job_sha256'] == record['sha256'] == digest(ROOT/record['path'])
        for name in ('cylinder_s02_terminal_tail.py', 'run_cylinder_s02_terminal_job.py',
                     'cylinder_s02_tail_v2.py'):
            assert config['source_sha256'][name] == seal['hashes']['scripts/'+name]
            assert digest(run/'sources'/name) == seal['hashes']['scripts/'+name]
        with np.load(run/'step_000000000/preview_trajectories.npz') as trace:
            for i in range(4):
                inactive = np.flatnonzero(~trace['active'][i])
                if len(inactive):
                    start = inactive[0]
                    assert start > 0
                    for name in ('qpos', 'qvel'):
                        tail = trace[name][i, start:]
                        np.testing.assert_array_equal(tail, np.broadcast_to(trace[name][i, start-1], tail.shape))
                    tails += 1
        if record['name'] == 'eval/B3_friction/ppo_s1002':
            with np.load(run/'recovery_prefix.npz') as actual, np.load(
                    OUT/'diagnostics/physics_failure_002/prefix.npz') as old:
                np.testing.assert_array_equal(actual['alive'], old['alive'])
                active = np.concatenate([np.ones((200, 1), bool), old['alive'][:, :-1]], axis=1)
                for name in ('qpos', 'qvel'):
                    np.testing.assert_array_equal(actual[name][active], old[name][active])
        checked.append(record['name'])
    result = dict(complete=not missing, checked=checked, missing=missing,
                  terminal_previews_checked=tails, amendment_sha256=digest(MANIFEST))
    (OUT/'terminal_tail_v2_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=not missing, checked=len(checked), missing=len(missing),
                         terminal_previews_checked=tails)), flush=True)
    if missing and not allow_partial:
        raise RuntimeError('Terminal-protected evaluations incomplete')


def run():
    seal = verify()
    allowed = {str(ROOT/r['path']) for r in seal['jobs']}
    import run_cylinder_s02 as controller

    def dispatch(command, **kwargs):
        verify()
        assert command[1] == str(ROOT/'scripts/run_cylinder_s02_job.py')
        assert command[2] == '--job' and command[3] in allowed
        routed = list(command)
        routed[1] = str(ROOT/'scripts/run_cylinder_s02_terminal_job.py')
        return subprocess.run(routed, **kwargs)

    # Reuse the original immutable lock, hashes, scheduling, and resume checks.
    # Only this module binding changes, not subprocess.run globally.
    controller.subprocess = SimpleNamespace(run=dispatch, STDOUT=subprocess.STDOUT)
    sys.argv[1:] = ['--phase', 'eval', '--workers', '2']
    controller.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'audit'))
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'audit':
        audit(args.allow_partial)
    else:
        run()
