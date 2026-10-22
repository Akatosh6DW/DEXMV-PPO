"""Use the exact routed residual inference with unchanged composition audit."""

import json
from pathlib import Path
import sys

import audit_s09_learned_base_residual as audit
import run_s09_learned_residual_experts as route
from run_s09_learned_residual_experts import ARCHITECTURE,residual_experts
from retarget_s09_reference import digest


if __name__ == "__main__":
    run = Path(sys.argv[sys.argv.index("--run")+1])
    cfg = json.loads((run/"run_config.json").read_text())
    if cfg.get("policy_architecture") != ARCHITECTURE:
        raise ValueError("This entry audits the six-residual-expert variant")
    assert digest(Path(route.__file__)) == cfg["sources"][Path(route.__file__).name]
    with residual_experts():
        audit.main()
    step = int(sys.argv[sys.argv.index("--step")+1]) if "--step" in sys.argv else 0
    path = run/f"step_{step:09d}/native_replay_audit.json"
    record = json.loads(path.read_text())
    record["residual_expert_audit_source_sha256"] = digest(Path(__file__))
    path.write_text(json.dumps(record,indent=2)+"\n")
