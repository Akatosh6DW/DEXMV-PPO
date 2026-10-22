"""Finalize the frozen experiment only after every scheduled job completes."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/orca_cylinder_s02'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    completion = OUT/'eval_completion.json'
    while not completion.exists():
        if not args.wait:
            raise RuntimeError('Full evaluation has not completed')
        # A disappeared controller is a failure to inspect, not silent waiting.
        with (OUT/'controller.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError('Controller is not running and evaluation is incomplete')
        time.sleep(30)
    protocol_hash = digest(OUT/'protocol.json')
    for phase, count in (('train', 50), ('eval', 388)):
        record = json.loads((OUT/(phase+'_completion.json')).read_text())
        if not record['complete'] or record['protocol_sha256'] != protocol_hash or len(record['records']) != count:
            raise ValueError('Invalid phase completion: '+phase)
    seal = json.loads((OUT/'analysis_manifest.json').read_text())
    if seal['protocol_sha256'] != protocol_hash:
        raise ValueError('Analysis plan belongs to another protocol')
    for name, expected in seal['hashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Sealed analysis changed: '+name)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_GL='egl', MPLCONFIGDIR='/tmp/cylinder_s02_matplotlib')
    commands = [
        ['audit_orca_suite.py', '--root', str(OUT)],
        ['report_cylinder_s02.py'],
        ['render_cylinder_s02.py'],
        ['audit_orca_suite.py', '--root', str(OUT), '--require-videos'],
        ['status_cylinder_s02.py'],
    ]
    if (OUT/'terminal_tail_amendment.json').exists():
        commands.insert(0, ['audit_cylinder_s02_tail.py'])
    if (OUT/'terminal_tail_v2.json').exists():
        commands.insert(0, ['cylinder_s02_tail_v2.py', 'audit'])
    if list((OUT/'numerical_recovery_plans').glob('*.json')):
        commands.insert(0, ['audit_cylinder_s02_numerical.py'])
    for index, (name, *arguments) in enumerate(commands):
        print('FINALIZE', name, flush=True)
        with (OUT/'logs'/('finalize_%02d_%s.log' % (index, name))).open('w') as stream:
            subprocess.run([sys.executable, str(ROOT/'scripts'/name), *arguments], cwd=ROOT,
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if name == 'report_cylinder_s02.py':
            for amendment in ('terminal_tail_amendment', 'terminal_tail_v2'):
                if (OUT/(amendment+'.json')).exists():
                    with (OUT/'report/results.md').open('a') as report:
                        report.write('\n'+(OUT/(amendment+'.md')).read_text())
            if list((OUT/'numerical_recovery_plans').glob('*.json')):
                with (OUT/'report/results.md').open('a') as report:
                    report.write('\n'+(OUT/'numerical_recovery.md').read_text())
    audit = json.loads((OUT/'audit.json').read_text())
    stats = json.loads((OUT/'report/statistics.json').read_text())
    if not audit['complete'] or not audit['videos_checked'] or not stats['complete']:
        raise ValueError('Final evidence incomplete')
    result = dict(complete=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=protocol_hash, audit_sha256=digest(OUT/'audit.json'),
                  statistics_sha256=digest(OUT/'report/statistics.json'),
                  videos_manifest_sha256=digest(OUT/'videos/manifest.json'),
                  report=str(OUT/'report/results.md'), videos=str(OUT/'videos/index.md'))
    for name in ('terminal_tail_audit', 'terminal_tail_v2_audit', 'numerical_recovery_audit'):
        if (OUT/(name+'.json')).exists():
            result[name+'_sha256'] = digest(OUT/(name+'.json'))
    (OUT/'completion.json').write_text(json.dumps(result, indent=2)+'\n')
    print('EXPERIMENT_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
