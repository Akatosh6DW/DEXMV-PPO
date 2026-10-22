"""Exercise cap interventions and both controls without using formal test resets."""

import argparse
import json
from pathlib import Path

from evaluate_oakink_cap import evaluate
from audit_oakink_cap_evaluation import audit
from oakink_cap_protocol import scenarios


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1] / "artifacts/orca_formal"
    run = root / "CAP31_auditable_residual_smoke"
    cases = [(run, scenario, baseline) for scenario in scenarios() for baseline in (None, "video", "initial")]
    cases += [(root / name, "nominal", None) for name in ("CAP35_auditable_bc_smoke", "CAP36_auditable_guided_initialization")]
    records = []
    for source, scenario, baseline in cases:
        name = source.name + "__" + scenario + "__" + (baseline or "policy")
        report = evaluate(source, scenario, args.output / name, baseline, development=True)
        checked = audit(args.output / name, dict(run=source.name, scenario=scenario, baseline=baseline), source.parent, development=True)
        (args.output / name / "independent_audit.json").write_text(json.dumps(checked, indent=2) + "\n")
        records.append(dict(label=name, success=report["metrics"]["success"]))
        (args.output / "progress.json").write_text(json.dumps(records, indent=2))
    (args.output / "complete.json").write_text(json.dumps(dict(passed=True, development=True, cases=len(records),
                                                               seeds=[19020, 19021]), indent=2))


if __name__ == "__main__":
    main()
