"""Descriptive paired results, held-out controls, and transparent repeat checks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_orca_residual import diagnose


def read_evaluation(run):
    meta = json.loads((run/"run_config.json").read_text())
    checkpoint = run/"step_000000000"
    episodes = json.loads((checkpoint/"episodes.json").read_text())
    with np.load(checkpoint/"trajectories.npz", allow_pickle=False) as z:
        final_hold = z["hold"][:, -1]
    return dict(success=float(np.mean(episodes["success30"])),
                success_count=int(np.sum(episodes["success30"])),
                episode_count=len(episodes["seeds"]),
                mean_best_hold=float(np.mean(episodes["hold_sec"])),
                mean_final_continuous_hold=float(final_hold.mean()),
                final_continuous30=float(np.mean(final_hold >= 30)),
                per_episode_success=episodes["success30"],
                seeds=episodes["seeds"],
                diagnostics=diagnose(checkpoint/"trajectories.npz", meta["bound"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--extras", type=Path, required=True)
    p.add_argument("--candidate-repeat", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    analysis = json.loads((args.suite/"analysis.json").read_text())
    if not analysis["protocol"]["confirmatory"] or analysis["protocol"]["seeds"] != [0, 1, 2]:
        raise ValueError("This report requires the registered three-seed confirmatory matrix")
    heldout = json.loads((args.evaluation/"results.json").read_text())
    extra = json.loads((args.extras/"results.json").read_text())
    records = []
    lookup = {}
    for row in heldout:
        record = dict(task=row["task"], arm=row["arm"], seed=row["seed"], kind=row["kind"],
                      checkpoint_step=row["checkpoint_step"], evaluation=row["evaluation"],
                      **read_evaluation(Path(row["evaluation"])))
        if row["checkpoint_step"] == 0:
            record["checkpoint_label"] = {"pure": "untrained zero-mean absolute controller",
                                           "bc": "BC-only; no PPO updates yet",
                                           "residual": "zero residual; prior-only controller"}[row["arm"]]
        else:
            record["checkpoint_label"] = "PPO-updated controller"
        records.append(record)
        lookup[(row["task"], row["arm"], row["seed"], row["kind"])] = record
    aggregates = []
    for item in analysis["aggregates"]:
        aggregate = dict(item)
        for kind in ("selected", "terminal"):
            group = [r for r in records if (r["task"], r["arm"], r["kind"]) ==
                     (item["task"], item["arm"], kind)]
            assert len(group) == item["n_training_seeds"]
            for metric in ("success", "mean_best_hold", "mean_final_continuous_hold", "final_continuous30"):
                aggregate[f"{kind}_{metric}"] = float(np.mean([r[metric] for r in group]))
            aggregate[f"{kind}_success_min"] = min(r["success"] for r in group)
            aggregate[f"{kind}_success_max"] = max(r["success"] for r in group)
            aggregate[f"{kind}_step0_count"] = sum(r["checkpoint_step"] == 0 for r in group)
        aggregates.append(aggregate)
    baselines, repeats = [], []
    for item in extra:
        measured = read_evaluation(Path(item["evaluation"]))
        if item["name"].startswith("prior_"):
            baselines.append(dict(task=item["name"].removeprefix("prior_"), **measured))
        else:
            meta = json.loads((Path(item["evaluation"])/"run_config.json").read_text())
            original = lookup[("cube", meta["arm"], 0, "selected")]
            assert original["seeds"] == measured["seeds"]
            repeats.append(dict(arm=meta["arm"], training_seed=0,
                                checkpoint_step=original["checkpoint_step"],
                                original_success=original["success"], repeat_success=measured["success"],
                                changed_episode_count=int(np.sum(np.array(original["per_episode_success"]) !=
                                                                  np.array(measured["per_episode_success"]))),
                                original=original["evaluation"], repeated=item["evaluation"]))
    differences = []
    for shape in analysis["protocol"]["shapes"]:
        for other in ("pure", "bc"):
            for kind in ("selected", "terminal"):
                diffs = [lookup[(shape, "residual", s, kind)]["success"]-
                         lookup[(shape, other, s, kind)]["success"] for s in analysis["protocol"]["seeds"]]
                differences.append(dict(task=shape, comparison=f"residual minus {other}", kind=kind,
                                        paired_differences=diffs, mean=float(np.mean(diffs)),
                                        positive_seed_count=sum(d > 0 for d in diffs),
                                        negative_seed_count=sum(d < 0 for d in diffs)))
    candidate_repeat = None
    if args.candidate_repeat:
        measured = read_evaluation(args.candidate_repeat)
        original = lookup[("cuboid", "residual", 1, "terminal")]
        assert original["seeds"] == measured["seeds"]
        candidate_repeat = dict(task="cuboid", arm="residual", training_seed=1,
                                checkpoint_step=original["checkpoint_step"],
                                original_success=original["success"], repeat_success=measured["success"],
                                changed_episode_count=int(np.sum(np.array(original["per_episode_success"]) !=
                                                                  np.array(measured["per_episode_success"]))),
                                original=original["evaluation"], repeated=str(args.candidate_repeat),
                                posthoc=True, excluded_from_aggregates=True)
    report = dict(aggregates=aggregates, heldout=records, prior_only=baselines,
                  posthoc_candidate_repeat=candidate_repeat,
                  repeatability=repeats, paired_differences=differences,
                  methodology=analysis["protocol"],
                  caveats=["Only3 independent paired training seeds, not96 independent training runs.",
                           "Primary success is any continuous30s hold plus valid final frame and no terminal drop; not necessarily a final30s uninterrupted interval.",
                           "Final uninterrupted30s rate is separately reported as final_continuous30.",
                           "All durations use20Hz control-frame samples and the shared0.01s force-cache offset.",
                           "Saved evaluation mean_return sums only nonterminal-frame rewards and excludes the terminal-transition reward; it is not the complete PPO training return and is not used for selection.",
                           "One cube video with geometric calibration; sphere/cuboid PPO retrained per task, not zero-shot universal grasping.",
                           "BC has additional selected simulated demonstrations and supervised updates; equal PPO steps are not equal total data/preparation cost.",
                           "Registered repeat checks use cube training seed0 selected checkpoints. All three selected step0 in this experiment, so these repeats test initial controllers, not PPO-updated policies.",
                           "The optional cuboid residual seed1 terminal repetition is a post-hoc candidate check, excluded from checkpoint selection and method aggregates.",
                           "Wall time includes compilation/cache effects; no unqualified speed or hardware-deployment claim."])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"report.json").write_text(json.dumps(report, indent=2))
    with (args.output/"comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps(dict(aggregates=aggregates, prior_only=baselines, repeatability=repeats), indent=2))


if __name__ == "__main__":
    main()
