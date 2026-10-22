"""Independently recheck the declared terminal-tail recovery evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/orca_cylinder_s02'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    amendment_path = OUT/'terminal_tail_amendment.json'
    amendment = json.loads(amendment_path.read_text())
    for name, expected in amendment['sources'].items():
        assert digest(ROOT/name) == expected, name
    assert amendment['protocol_sha256'] == digest(OUT/'protocol.json')
    diagnostic = OUT/'diagnostics/physics_failure_001'
    validation = OUT/'diagnostics/terminal_tail_validation'
    assert amendment['diagnosis_sha256'] == digest(diagnostic/'diagnosis.json')
    assert amendment['validation_sha256'] == digest(validation/'validation.json')
    run = OUT/amendment['job']
    config = json.loads((run/'run_config.json').read_text())
    assert config['job_sha256'] == amendment['job_sha256']
    assert config['evaluation_recovery']['amendment_sha256'] == digest(amendment_path)
    for name, expected in amendment['sources'].items():
        if name.endswith('audit_cylinder_s02_tail.py'):
            continue
        assert config['source_sha256'][Path(name).name] == expected
    evidence = json.loads((run/'recovery_evidence.json').read_text())
    assert evidence['passed'] and evidence['amendment_sha256'] == digest(amendment_path)
    assert evidence['prefix_sha256'] == digest(run/'recovery_prefix.npz')
    with np.load(run/'recovery_prefix.npz') as actual, np.load(diagnostic/'prefix.npz') as old:
        np.testing.assert_array_equal(actual['alive'], old['alive'])
        active = np.concatenate([np.ones((len(old['seeds']), 1), bool), old['alive'][:, :-1]], axis=1)
        for name in ('qpos', 'qvel'):
            np.testing.assert_array_equal(actual[name][active], old[name][active])
    episodes = json.loads((run/'step_000000000/episodes.json').read_text())
    assert episodes == json.loads((validation/'B3_friction_ppo_s1001_episodes.json').read_text())
    diagnosis = json.loads((diagnostic/'diagnosis.json').read_text())
    assert diagnosis['already_terminated'] and not episodes[diagnosis['index'][0]]['clean_success30']
    result = dict(passed=True, job=amendment['job'], episodes=200,
                  active_prefix_exact=True, amendment_sha256=digest(amendment_path))
    (OUT/'terminal_tail_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
