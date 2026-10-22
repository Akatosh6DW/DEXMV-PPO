"""Audit phase-handoff actions and native physics with the unchanged tolerances."""

import argparse
import json
from pathlib import Path

import audit_s09_bc_repair
from run_s09_phase_handoff import ARCHITECTURE, handoff_networks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    cfg = json.loads((args.run/"run_config.json").read_text())
    if cfg.get("policy_architecture") != ARCHITECTURE:
        raise ValueError("Not a phase-handoff controller")
    with handoff_networks():
        audit_s09_bc_repair.main()


if __name__ == "__main__":
    main()
