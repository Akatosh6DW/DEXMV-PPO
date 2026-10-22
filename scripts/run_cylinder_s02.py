"""Run the full frozen cylinder S02 matrix, with bounded concurrency and resume checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/orca_cylinder_s02'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('train', 'eval', 'all'), default='all')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    protocol_path = OUT/'protocol.json'
    protocol_hash = digest(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    (OUT/'logs').mkdir(exist_ok=True)
    lock = threading.Lock()

    def verify():
        if digest(protocol_path) != protocol_hash:
            raise ValueError('Protocol changed during execution')
        for name, expected in protocol['source_hashes'].items():
            if digest(ROOT/name) != expected:
                raise ValueError('Frozen input changed: '+name)

    def run(record):
        path = ROOT/record['path']
        if digest(path) != record['sha256']:
            raise ValueError('Job changed')
        job = json.loads(path.read_text())
        out = ROOT/job['output']
        summary = out/'summary.json'
        if summary.exists():
            result = json.loads(summary.read_text())
            if result['job_sha256'] != digest(path) or result['status'] != 'completed':
                raise ValueError('Invalid previous completion')
            return dict(name=record['name'], skipped=True, summary=result)
        if out.exists():
            raise FileExistsError('Inspect incomplete job before restarting: '+str(out))
        if shutil.disk_usage(OUT).free < 600*1024**2:
            raise RuntimeError('Less than 600 MiB free; stopping without deleting prior experiments')
        verify()
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   XLA_FLAGS='--xla_gpu_exclude_nondeterministic_ops=true')
        log = OUT/'logs'/(path.stem+'.log')
        print('START', record['name'], flush=True)
        with log.open('x') as stream:
            process = subprocess.run([sys.executable, str(ROOT/'scripts/run_cylinder_s02_job.py'),
                                      '--job', str(path)], cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode or not summary.exists():
            raise RuntimeError('Job failed; inspect '+str(log))
        result = json.loads(summary.read_text())
        if result['job_sha256'] != record['sha256'] or result['final_step'] != job['steps']:
            raise ValueError('Unexpected job completion/step count')
        verify()
        entry = dict(name=record['name'], skipped=False, summary=result)
        with lock, (OUT/'execution.jsonl').open('a') as stream:
            stream.write(json.dumps(entry)+'\n')
        print('DONE', json.dumps(entry), flush=True)
        return entry

    with (OUT/'controller.lock').open('a') as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify()
        for phase in ('train', 'eval') if args.phase == 'all' else (args.phase,):
            records = [r for r in protocol['jobs'] if r['name'].startswith(phase+'/')]
            results = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run, record) for record in records]
                try:
                    for future in as_completed(futures):
                        results.append(future.result())
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            (OUT/(phase+'_completion.json')).write_text(json.dumps(dict(
                complete=True, protocol_sha256=protocol_hash, records=results), indent=2)+'\n')
            print('PHASE_COMPLETE', phase, len(results), flush=True)


if __name__ == '__main__':
    main()
