"""Publish readable cube-only tables without changing frozen scores or recipes."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


SEEDS = (301, 302, 303, 304, 305)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(data, original, audit):
    if data != original:
        raise ValueError("Recomputed statistics differ from audited published statistics")
    if not data["evaluations_complete"] or data["missing"] or data["evaluations"] != 513:
        raise ValueError("All 513 cube evaluations are required")
    if not data["learning"]["complete"] or len(data["learning"]["runs"]) != 80:
        raise ValueError("All 80 cube training records are required")
    if len(data["table"]) != 102 or len(data["paired_effects"]) != 102:
        raise ValueError("Incomplete condition/contrast coverage")
    for row in data["table"]:
        if not row["complete_seeds"] or set(map(int, row["invalid_initial_counts"])) != set(SEEDS):
            raise ValueError("Incomplete training seeds")
        for metric in row["metrics"].values():
            if metric["n_training_seeds"] != 5 or len(metric["individual"]) != 5:
                raise ValueError("Each estimate must have five training seeds")
    if len(data["failure_diagnostics"]) != 513:
        raise ValueError("Incomplete failure diagnostics")
    for failure in data["failure_diagnostics"].values():
        if len(failure["episodes"]) != 64 or Counter(e["reason"] for e in failure["episodes"]) != failure["counts"]:
            raise ValueError("Failure counts do not reconcile with 64 episodes")
    cube = audit["tasks"]["cube"]
    if not cube["machine_checks_passed"] or len(cube["training"]) != 80 or len(cube["evaluations"]) != 513:
        raise ValueError("Missing complete cube machine audit")
    if not cube["published_statistics_match"] or not cube["videos"]["passed"]:
        raise ValueError("Cube statistics/video audit did not pass")


def metric_rows(data):
    aggregate, individual = [], []
    for row in data["table"]:
        for name, value in row["metrics"].items():
            common = dict(condition=row["condition"], scenario=row["scenario"], metric=name)
            aggregate.append(dict(**common, mean=value["mean"], sd=value["std"],
                                  ci95_low=value["bootstrap95"][0], ci95_high=value["bootstrap95"][1],
                                  training_seeds=value["n_training_seeds"]))
            for seed, number in zip(SEEDS, value["individual"]):
                individual.append(dict(**common, train_seed=seed, value=number,
                                       invalid_initial_count=row["invalid_initial_counts"][str(seed)]))
    return aggregate, individual


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(str(x) for x in row) + " |" for row in rows]


def estimate(value, scale=1):
    return f"{scale * value['mean']:.2f} +/- {scale * value['std']:.2f}"


def interval(value, scale=1):
    return f"[{scale * value['bootstrap95'][0]:.2f}, {scale * value['bootstrap95'][1]:.2f}]"


def export(statistics, original, audit_path):
    data = read(statistics)
    validate(data, read(original), read(audit_path))
    output = statistics.parent
    aggregates, individuals = metric_rows(data)
    pairs = []
    for row in data["paired_effects"]:
        effect = row["effect"]
        pairs.append(dict(kind=row["kind"], scenario=row["scenario"], left=row["left"], right=row["right"],
                          mean_pp=100 * effect["mean"], sd_pp=100 * effect["std"],
                          ci95_low_pp=100 * effect["bootstrap95"][0],
                          ci95_high_pp=100 * effect["bootstrap95"][1],
                          exact_two_sided_p=effect["exact_two_sided_signflip_p"],
                          **{f"seed{s}_pp": 100 * v for s, v in zip(effect["seeds"], effect["individual"])}))
    learning = [{k: v for k, v in row.items() if k not in ("steps", "success")}
                for row in data["learning"]["runs"]]
    curves = [dict(run=row["run"], step=step, development_success30=success)
              for row in data["learning"]["runs"] for step, success in zip(row["steps"], row["success"])]
    episodes = [dict(evaluation=label, reset_seed=25000 + i, **episode)
                for label, failure in sorted(data["failure_diagnostics"].items())
                for i, episode in enumerate(failure["episodes"])]
    reasons = sorted({e["reason"] for e in episodes})
    failures = [dict(evaluation=label, **{r: failure["counts"].get(r, 0) for r in reasons})
                for label, failure in sorted(data["failure_diagnostics"].items())]
    zero = [dict(condition=k, **v) for k, v in sorted(data["zero_correction"].items())]
    initial_quality = []
    for row in data["table"]:
        for seed, rate in zip(SEEDS, row["metrics"]["success30"]["individual"]):
            invalid = row["invalid_initial_counts"][str(seed)]
            successes = round(rate * 64)
            valid = 64 - invalid
            initial_quality.append(dict(condition=row["condition"], scenario=row["scenario"], train_seed=seed,
                                        invalid_initial_count=invalid, valid_initial_count=valid,
                                        success30_count=successes, strict_success30=rate,
                                        success30_valid_initial=successes / valid if valid else None))
    files = dict(metrics=aggregates, per_seed=individuals, paired_effects=pairs, learning=learning,
                 learning_curves=curves, failures=failures, failure_episodes=episodes, zero_controls=zero,
                 initial_quality=initial_quality)
    for name, rows in files.items():
        write_csv(output / (name + ".csv"), rows)

    lines = ["# Cube-Only Complete Statistics", "",
             "Scope: source-matched 53 mm HIC20 cube. No cap experiment is included or required.",
             "80 training jobs; 513/513 held-out evaluations; 64 episodes each; 32832 episode executions.",
             "Counts reuse paired initial states across methods/scenarios; they are not independent training seeds.",
             "[Chinese reading guide](../../../docs/orca_cube_complete_statistics.md)", "",
             "## Statistical Definitions", "",
             "Primary success: longest valid unsupported hold >=30 s AND valid final state. Invalid initial states count as failures.",
             "All learned-policy groups have training seeds 301..305; each policy uses reset seeds 25000..25063.",
             "Mean +/- sample SD and bootstrap95 resample training seeds, conditional on the fixed test set.",
             "Five seeds are limited; degenerate intervals do not establish population certainty.",
             "Exact two-sided paired sign-flip p-values have minimum 0.0625. Secondary tests are unadjusted/exploratory.",
             "Each seed uses its fixed final checkpoint; no test-based model selection or retraining was performed.",
             "Peak force: maximum summed normal force on one finger over all active saved frames/episodes for a seed, then averaged across seeds.",
             "Penetration: per-seed maximum in mm, then averaged; neither statistic establishes hardware safety.",
             "Frames are sampled at 20 Hz, not every physics substep. Mean hold is the mean longest valid hold, not terminal hold.",
             "initial_quality.csv additionally separates invalid resets from valid-reset control failures; the primary denominator remains 64.",
             "Full-precision CSV values use native units: success proportions 0..1, seconds, N, mm; paired effects use percentage points.", "",
             "## Files", ""]
    lines += [f"- [{name}.csv]({name}.csv): {len(rows)} rows" for name, rows in files.items()]
    lines += ["- [Full structured statistics](statistics.json)",
              "- [Development curves](development_curves.png), not held-out performance.", "",
              "## All Conditions", "",
              "Every predefined scenario is shown, including failures and ablations. No pooling of the old 36 mm study."]
    scenarios = ["nominal"] + sorted({r["scenario"] for r in data["table"]} - {"nominal"})
    for scenario in scenarios:
        selected = [r for r in data["table"] if r["scenario"] == scenario]
        lines += ["", f"### {scenario}", ""]
        rows = []
        for row in selected:
            m = row["metrics"]
            rows.append([row["condition"], estimate(m["success30"], 100), interval(m["success30"], 100),
                         f"{100*m['success15']['mean']:.2f}", f"{100*m['valid_final']['mean']:.2f}",
                         estimate(m["strict_mean_hold"]), estimate(m["peak_force_n"]),
                         estimate(m["max_penetration_mm"]), f"{100*m['acquisition_rate']['mean']:.2f}",
                         ",".join(str(row["invalid_initial_counts"][str(s)]) for s in SEEDS)])
        lines += md_table(["Condition", "30s success % +/- SD", "95% CI %", "15s+final %", "Valid final %",
                           "Hold s +/- SD", "Peak N +/- SD", "Penetration mm +/- SD", "Acquired %", "Invalid /64 by seed"], rows)
    lines += ["", "## Every Training Seed", "", "Success30 percentages, in seed order. Other per-seed metrics are in per_seed.csv.", ""]
    lines += md_table(["Condition", "Scenario", *map(str, SEEDS)], [
        [r["condition"], r["scenario"], *[f"{100*v:.4f}" for v in r["metrics"]["success30"]["individual"]]]
        for r in data["table"]])
    lines += ["", "## All Paired Contrasts", "", "Direction is LEFT minus RIGHT, in percentage points. Not multiplicity-adjusted.", ""]
    lines += md_table(["Kind", "Scenario", "Left", "Right", "Difference pp +/- SD", "95% CI pp", "Exact p"], [
        [r["kind"], r["scenario"], r["left"], r["right"], estimate(r["effect"], 100), interval(r["effect"], 100),
         f"{r['effect']['exact_two_sided_signflip_p']:.4f}"] for r in data["paired_effects"]])
    lines += ["", "## Zero-Correction Controls", "", "One deterministic controller per condition, 64 resets; no five-training-seed confidence interval.", ""]
    lines += md_table(["Condition", "30s+final %", "15s+final %", "Mean longest hold s", "Max finger N"], [
        [r["condition"], f"{100*r['success30']:.2f}", f"{100*r['success15']:.2f}",
         f"{r['strict_mean_hold']:.4f}", f"{r['peak_force_n']:.4f}"] for r in zero])
    lines += ["", "## All Training Curves", "",
              "Development monitoring only. AUC is normalized by PPO steps; BC's single point is not an RL learning-speed measurement.",
              "The 75% step is the FIRST of two consecutive recorded checkpoints >=75%, not permanent success.",
              "NA is right-censored, never zero training steps. BC fit time is not added twice for a paired guided run.", ""]
    lines += md_table(["Run", "First %", "Last %", "Max %", "AUC", "Regression pp", "75% pair start", "BC fit s", "Monitoring elapsed s"], [
        [r["run"], f"{100*r['first']:.2f}", f"{100*r['last']:.2f}", f"{100*r['maximum']:.2f}",
         f"{r['normalized_auc']:.6f}", f"{100*r['peak_to_final_regression']:.2f}",
         r["first_two_checkpoints_at_least75_step"] if not r["threshold_not_reached"] else "NA",
         f"{r['bc_fit_seconds']:.3f}", f"{r['monitoring_elapsed_seconds']:.3f}"] for r in learning])
    lines += ["", "## Failure Counts by Condition", "",
              "Counts are mutually exclusive descriptive outcomes, not unique causal explanations. Termination does not automatically mean dropping.",
              "Each learned-policy row totals 320 episodes. Zero controls total 64. No cross-condition pooled success rate.", ""]
    grouped = {}
    for row in failures:
        label = row["evaluation"]
        key = label.split("_", 1)[1] if label.startswith("s") else label
        grouped.setdefault(key, Counter()).update({r: row[r] for r in reasons})
    lines += md_table(["Condition__scenario", *reasons], [[k, *[v[r] for r in reasons]] for k, v in sorted(grouped.items())])
    lines += ["", "## Provenance", "", f"- Recomputed statistics SHA256: `{digest(statistics)}`",
              f"- Historical whole-study audit SHA256: `{digest(audit_path)}`; only its cube section is used here.",
              "- Historical statistics.entire_goal_complete/note fields retain the old whole-study scope; they are not cube coverage flags.",
              "- No environment, reward, checkpoints, threshold, or frozen evaluator was changed for this export.", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = dict(scope="cube_only", evaluations=513, episode_executions=len(episodes), training_jobs=80,
                    source_statistics_sha256=digest(statistics), original_statistics_sha256=digest(original),
                    audit_sha256=digest(audit_path), exporter_sha256=digest(Path(__file__)),
                    rows={name: len(rows) for name, rows in files.items()},
                    files={name: digest(output / name) for name in ["results.md", *[n + ".csv" for n in files]]})
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.statistics, args.original, args.audit)
    print(json.dumps(result["rows"]))


if __name__ == "__main__":
    main()
