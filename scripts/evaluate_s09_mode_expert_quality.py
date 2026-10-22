"""Full-case contact/failure diagnostics for the separately labelled experts."""

import json
from pathlib import Path
import sys

from run_s09_mode_experts import ARCHITECTURE, expert_networks
import evaluate_s09_failure_quality as quality


def main():
    argv = sys.argv[1:]
    if "--steps" not in argv or int(argv[argv.index("--steps")+1]) != 0:
        raise ValueError("This is an evaluation-only entry")
    if argv[argv.index("--method")+1] != "guided":
        raise ValueError("The expert variant supplies full joint targets")
    checkpoint = Path(argv[argv.index("--checkpoint")+1])
    parent = checkpoint.parent
    if not (parent/"run_config.json").exists():
        parent = parent.parent
    cfg = json.loads((parent/"run_config.json").read_text())
    if cfg.get("policy_architecture") != ARCHITECTURE:
        raise ValueError("Use an explicit expert checkpoint")
    output = Path(argv[argv.index("--output")+1])
    try:
        with expert_networks():
            quality.main()
    finally:
        for name, field in (("run_config.json", "policy_architecture"),
                            ("improvement_config.json", "action_architecture")):
            path = output/name
            if path.exists():
                data = json.loads(path.read_text())
                data[field] = ARCHITECTURE
                path.write_text(json.dumps(data, indent=2)+"\n")


if __name__ == "__main__":
    main()
