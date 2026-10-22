"""Exercise all evaluation interventions on development resets, never test seeds."""

import argparse
import json
from pathlib import Path

import jax

from audit_orca_formal_evaluation import audit
from evaluate_orca_formal import evaluate
from orca_formal_protocol import scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        parser.error("Use JAX_PLATFORMS=cpu")
    root = Path(__file__).resolve().parents[1]
    residual = root / "artifacts/orca_collision_refit/R15_hic53_native_bound15_seed202"
    jobs = [(residual, 655360, name, None) for name in scenarios()]
    jobs += [(residual, 655360, "nominal", mode) for mode in ("video", "endpoint", "initial")]
    jobs += [(root / "artifacts/orca_formal/F01" / f"s301_{method}_off",
              0 if method == "bc" else 655360, "nominal", None) for method in ("bc", "ppo", "guided")]
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for run, step, scenario, zero in jobs:
        name = run.name + "__" + scenario + ("__zero_" + zero if zero else "")
        destination = args.output / name
        evaluate(run, step, scenario, destination, [19000, 19001], zero, development=True)
        job = dict(run=run.name, step=step, scenario=scenario, zero=zero)
        checked = audit(destination, job, run.parent, development=True)
        (destination / "independent_audit.json").write_text(json.dumps(checked, indent=2) + "\n")
        records.append(dict(label=name, passed=checked["passed"], report_sha256=checked["report_sha256"]))
        (args.output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        print("AUDIT_PASS", name, flush=True)
    (args.output / "complete.json").write_text(json.dumps(dict(development_only=True, cases=len(records),
                                                             seeds=[19000, 19001], all_passed=True), indent=2) + "\n")


if __name__ == "__main__":
    main()
