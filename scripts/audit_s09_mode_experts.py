"""Use expert inference with the unchanged full-target/native-physics audit."""

import json
from pathlib import Path
import sys

from run_s09_mode_experts import ARCHITECTURE, expert_networks
import audit_s09_bc_repair


if __name__ == "__main__":
    run = Path(sys.argv[sys.argv.index("--run")+1])
    cfg = json.loads((run/"run_config.json").read_text())
    if cfg.get("policy_architecture") != ARCHITECTURE:
        raise ValueError("Use this audit only for the explicit expert variant")
    with expert_networks():
        audit_s09_bc_repair.main()
