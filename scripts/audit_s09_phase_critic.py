"""Verify the original residual action/physics under a phase-critic entry."""

import json
from pathlib import Path
import sys

import audit_s09_learned_base_residual as audit
import run_s09_phase_critic as route
from retarget_s09_reference import digest


def main():
    run = Path(sys.argv[sys.argv.index("--run")+1])
    cfg = json.loads((run/"run_config.json").read_text())
    protocol = json.loads((run/"phase_critic_config.json").read_text())
    for module in (route,route.experts):
        assert digest(Path(module.__file__)) == cfg["sources"][Path(module.__file__).name]
    assert protocol["source_sha256"] == digest(Path(route.__file__))
    assert protocol["initial_critics_identical"] and cfg["env_config"]["release_sec"] == 3
    with route.phase_critics(protocol["routing"]=="phase"),route.experts.residual_experts():
        audit.main()
    step = int(sys.argv[sys.argv.index("--step")+1]) if "--step" in sys.argv else 0
    path = run/f"step_{step:09d}/native_replay_audit.json"
    record = json.loads(path.read_text())
    record["phase_critic_audit_source_sha256"] = digest(Path(__file__))
    record["critic_routing"] = protocol["routing"]
    path.write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    main()
