"""Run the existing same-state recovery diagnostic with explicit expert actors."""

import json
from pathlib import Path
import sys

import probe_s09_recovery_student as probe
from run_s09_mode_experts import ARCHITECTURE, expert_networks


if __name__ == "__main__":
    start = sys.argv.index("--runs")+1
    for name in sys.argv[start:]:
        if name.startswith("--"):
            break
        cfg = json.loads((Path(name)/"run_config.json").read_text())
        if cfg.get("policy_architecture") != ARCHITECTURE:
            raise ValueError("Use only explicit six-expert runs")
    with expert_networks():
        probe.main()
