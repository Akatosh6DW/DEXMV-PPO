"""Audit the explicit deterministic phase-gating intervention."""

import json
from pathlib import Path
import sys

import audit_s09_learned_base_residual as audit
import run_s09_residual_phase_probe as probe
from retarget_s09_reference import digest


if __name__ == "__main__":
    run = Path(sys.argv[sys.argv.index("--run")+1])
    cfg = json.loads((run/"run_config.json").read_text())
    intervention = json.loads((run/"phase_probe.json").read_text())
    assert cfg["steps"] == 0 and cfg["policy_architecture"] == probe.ARCHITECTURE
    assert digest(Path(probe.__file__)) == cfg["sources"][Path(probe.__file__).name]
    with probe.phase_policy(intervention["phase"]):
        audit.main()
    path = run/"step_000000000/native_replay_audit.json"
    record = json.loads(path.read_text())
    record["phase_intervention_sha256"] = digest(run/"phase_probe.json")
    record["phase_audit_source_sha256"] = digest(Path(__file__))
    path.write_text(json.dumps(record,indent=2)+"\n")
