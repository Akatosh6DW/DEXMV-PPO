"""Clone all S02 jobs, changing only object/data bindings and output paths."""

import json
from pathlib import Path
import shutil

from prepare_cylinder_s02_source import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/orca_cylinder_s02'
ORIGINAL = ROOT/'artifacts/orca_benchmark_suite/S02/protocol.json'


def main():
    original = json.loads(ORIGINAL.read_text())
    for path, expected in original['source_hashes'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Original S02 input changed: '+path)
    manifest = json.loads((OUT/'inputs/demos/dataset_manifest.json').read_text())
    for name, path in (('model', 'model.xml'), ('reference', 'reference.npz'),
                       ('dataset', 'demos/trajectories.npz')):
        if digest(OUT/'inputs'/path) != manifest[name+'_sha256']:
            raise ValueError('Demonstration binding mismatch: '+name)
    if manifest['accepted'] < 4:
        raise ValueError('No valid BC dataset')
    protocol_path = OUT/'protocol.json'
    if protocol_path.exists():
        raise FileExistsError(protocol_path)
    jobs_dir = OUT/'jobs'
    jobs_dir.mkdir(exist_ok=False)
    records = []
    for item in original['jobs']:
        path = ROOT/item['path']
        if digest(path) != item['sha256']:
            raise ValueError('Original job changed')
        job = json.loads(path.read_text())
        for field in ('output', 'checkpoint'):
            if field in job:
                job[field] = job[field].replace(original['output'], str(OUT.relative_to(ROOT)), 1)
        new = jobs_dir/path.name
        new.write_text(json.dumps(job, indent=2)+'\n')
        records.append(dict(name=item['name'], path=str(new.relative_to(ROOT)), sha256=digest(new)))
    protocol = dict(original)
    protocol.update(version='CYLINDER_S02', date='2026-10-09',
                    output=str(OUT.relative_to(ROOT)), jobs=records,
                    original_protocol_sha256=digest(ORIGINAL),
                    original_source_hashes=original['source_hashes'],
                    source_hashes={}, source_selection='DexMV processed tomato_soup_can demonstration; see retarget_report',
                    changes=['53 mm cube replaced with 53 mm diameter/height upright cylinder, same mass',
                             'New published can demonstration, geometric cross-hand retargeting',
                             'Physical distal finger capsule centers used for contact correspondence',
                             'New successful replay BC dataset; all original learning methods unchanged'],
                    preregistration='docs/experiments/cylinder_s02_execution.md')
    paths = [ROOT/p for p in original['source_hashes']]
    paths += list((OUT/'inputs').glob('*.xml')) + list((OUT/'inputs').glob('*.npz'))
    paths += [OUT/'inputs/retarget_report.json', OUT/'inputs/demos/trajectories.npz',
              OUT/'inputs/demos/dataset_manifest.json', OUT/'source/selection96/selected_segment.npz',
              OUT/'source/selection96/selection.json', OUT/'source/processed/manifest.json',
              OUT/'source/fk_crosscheck.json', OUT/'source/processed/seq_096.npz']
    names = ('prepare_cylinder_s02_source.py', 'select_cylinder_s02_source.py',
             'retarget_cylinder_s02.py', 'orca_cylinder_s02_env.py', 'collect_cylinder_s02.py',
             'orca_cylinder_evaluation.py', 'run_cylinder_s02_job.py', 'prepare_cylinder_s02.py',
             'run_cylinder_s02.py', 'test_cylinder_s02.py', 'orca_shape_pilot_env.py',
             'check_dexmv_can_fk.py', 'report_orca_suite.py')
    paths += [ROOT/'scripts'/name for name in names]
    snapshot = OUT/'frozen_sources'
    snapshot.mkdir()
    for path in paths:
        relative = path.relative_to(ROOT)
        protocol['source_hashes'][str(relative)] = digest(path)
        if path.suffix == '.py':
            target = snapshot/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copyfile(ORIGINAL.parent/'statistics_plan.json', OUT/'statistics_plan.json')
    print(json.dumps(dict(jobs=len(records), training=50, evaluation=388,
                         formal_episodes=77600, protocol_sha256=digest(protocol_path))))


if __name__ == '__main__':
    main()
