"""Add per-gate diagnostics for every evaluated episode, not just previews."""

import json
from pathlib import Path
import sys

import evaluate_s09_quality as quality
from diagnose_s09_failure_modes import episode_reasons
from retarget_s09_reference import digest


def add_reasons(trace, episodes, object_qpos_adr):
    result = []
    for i, episode in enumerate(episodes):
        one = {k:v[i] for k,v in trace.items() if k != "diagnostic_initial_qpos"}
        reasons = episode_reasons(one, trace["diagnostic_initial_qpos"][i], object_qpos_adr)
        if reasons["success30"] != episode["success30"]:
            raise ValueError("Diagnostic labels disagree with the primary evaluator")
        result.append(dict(episode, failure_gates=reasons))
    return result


def main():
    runner = quality.improvement.runner
    original_evaluator, original_summarize = runner.evaluator, runner.summarize
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    object_address = []

    def evaluator(env, net, reference=False):
        evaluate = original_evaluator(env, net, reference=reference)
        object_address.append(env.oid)

        def with_initial(params, seeds):
            # Add data after the original compiled rollout; never change physics
            # or recompile its arithmetic solely for this diagnostic.
            initial, trace = evaluate(params, seeds)
            return initial, dict(trace, diagnostic_initial_qpos=initial)

        return with_initial

    def summarize(trace, seeds):
        row, episodes = original_summarize(trace, seeds)
        return row, add_reasons(trace, episodes, object_address[0])

    runner.evaluator, runner.summarize = evaluator, summarize
    try:
        quality.main()
    finally:
        runner.evaluator, runner.summarize = original_evaluator, original_summarize
        if (output / "sources").exists():
            metadata = dict(scope="All evaluated episodes, active released frames",
                gate_counts_overlap=True, original_success_unchanged=True,
                rollout_compilation_unchanged=True,
                boundary_disagreements_reported=True,
                source_sha256=digest(output / "sources" / Path(__file__).name))
            (output / "failure_config.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
