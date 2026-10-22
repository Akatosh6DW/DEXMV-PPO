"""Recheck both frozen studies; never infer whole-goal completion from counters."""

import argparse
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect_reports(roots, jobs):
    expected = {job["label"]: job for job in jobs}
    require(len(expected) == len(jobs), "Duplicate expected job")
    found = {}
    for root in roots:
        for path in sorted(root.glob("*/report.json")):
            label = path.parent.name
            require(label in expected, "Unexpected evaluation: " + str(path))
            require(label not in found, "Duplicate evaluation: " + label)
            found[label] = path.parent
    return found, sorted(expected.keys() - found.keys())


def contact_disposition(run):
    result = read(run / "formal_result.json")
    if result["penetration_max_mm"] <= 5 and result["forbidden_fraction"] <= .01:
        return "below_inspection_threshold"
    review = read(run / "contact_review.json")
    require(review["review_complete"] is True, "Contact review unfinished")
    require(review["disposition"] == "retain_scored_negative_episodes_continue_unchanged_protocol",
            "Unrecognized contact disposition")
    for key, filename in (("result_sha256", "formal_result.json"),
                          ("inspection_sha256", "contact_inspection.json")):
        require(review[key] == digest(run / filename), "Stale contact review: " + key)
    require(review["no_score_or_checkpoint_replacement"] is True
            and review["no_environment_reward_or_training_change"] is True,
            "Contact review changed the formal experiment")
    return review["disposition"]


def check_manifest(root, suite, expected):
    manifest = read(root / "manifest.json")
    require(Path(manifest["suite"]).resolve() == suite.resolve(), "Wrong evaluation suite")
    require(manifest["protocol_sha256"] == digest(suite / "protocol.json"), "Stale evaluator protocol")
    labels = [job["label"] for job in manifest["jobs"]]
    require(len(labels) == len(set(labels)), "Duplicate evaluator manifest job")
    for job in manifest["jobs"]:
        require(job == expected.get(job["label"]), "Evaluator manifest recipe mismatch")
    for name, sha in manifest["sources"].items():
        require(digest(root / "sources" / name) == sha == digest(ROOT / "scripts" / name),
                "Evaluator source drift: " + name)
    if "amendment_sha256" in manifest:
        require(digest(ROOT / "docs/orca_formal_evaluation_amendment.md") == manifest["amendment_sha256"],
                "Evaluation amendment changed")
    return set(labels)


def check_videos(video_root, reports, task):
    from render_orca_formal import select_cases as cube_cases, verify_video
    from render_oakink_cap_formal import select_cases as cap_cases

    selected = (cube_cases if task == "cube" else cap_cases)(reports)
    require(read(video_root / "selection.json") == selected, "Video selection differs from fixed rule")
    manifest = read(video_root / "manifest.json")
    expected = {(c["run"], c["episode"], az): c for c in selected
                for az in ((225,) if task == "cube" else (225, 315))}
    seen, evidence = set(), []
    for case in manifest["cases"]:
        key = (case["run"], case["episode"], case.get("azimuth", 225))
        require(key in expected and key not in seen, "Unexpected/duplicate video case")
        require(case["reason"] == expected[key]["reason"], "Video case label changed")
        seen.add(key)
        report = reports[case["run"]]
        outcome = (report["episodes"]["success30"] if task == "cube"
                   else report["independent_audit"]["episodes"]["success"])[case["episode"]]
        require(case["success30" if task == "cube" else "success"] == outcome, "Video outcome mislabelled")
        directory = Path(case["output"])
        rendering = read(directory / "report.json")
        require(rendering["trace_sha256"] == report["trace_sha256"]
                and rendering["episode"] == case["episode"], "Wrong rendered trajectory")
        require(digest(Path(rendering["trace"])) == report["trace_sha256"], "Rendered trace changed")
        model_key = "model_xml_sha256" if task == "cube" else "model_source_sha256"
        require(digest(Path(rendering["source_model"])) == report[model_key], "Wrong rendered model")
        require(rendering["renderer_source_sha256"] == digest(ROOT / "scripts/render_orca_modes.py"),
                "Renderer source changed")
        require(rendering["camera"] == dict(azimuth=key[2], elevation=-18,
                lookat=[.035, .025, .245], distance=.38 if task == "cube" else .42), "Wrong video camera")
        require(rendering["no_physics_resimulation"] is True
                and rendering["same_saved_states_both_modes"] is True
                and rendering["physics_arrays_match"] is True, "Video changed physics")
        for mode in ("stl", "collision"):
            path = directory / (mode + ".mp4")
            actual = verify_video(path, rendering)
            require(actual == case["videos"][mode], "Video file or ffprobe evidence changed")
            evidence.append(dict(path=str(path.resolve()), sha256=actual["sha256"]))
    require(seen == expected.keys() and manifest["complete"] is True, "Missing required videos")
    return dict(passed=True, cases=len(seen), videos=evidence,
                note="Hash/frame/selection verification only. Actual visual inspection remains mandatory.")


def audit_task(task, suite, roots, video_root, output):
    if task == "cube":
        from orca_formal_protocol import recipes
        from run_orca_formal import verify_frozen
        from run_orca_formal_evaluations import jobs as cube_jobs
        from audit_orca_formal_run import audit_run as train_audit
        from audit_orca_formal_evaluation import audit as eval_audit
        from analyze_orca_formal import analyze
        protocol = read(suite / "protocol.json")
        verify_frozen(protocol)
        jobs = cube_jobs("all")
    else:
        from oakink_cap_protocol import recipes, check_protocol
        from run_oakink_cap_evaluations import jobs as cap_jobs
        from audit_oakink_cap_run import audit_run as train_audit
        from audit_oakink_cap_evaluation import audit as eval_audit
        from analyze_oakink_cap import analyze
        protocol = check_protocol(suite / "protocol.json")
        jobs = cap_jobs()
    specifications = recipes()
    require(protocol["recipes"] == specifications, "Frozen recipe list differs from implementation")
    require(len(specifications) == (80 if task == "cube" else 20), "Training scope changed")
    require(len(jobs) == (513 if task == "cube" else 110), "Evaluation scope changed")
    training, missing_training = {}, []
    for spec in specifications:
        run = suite / spec["label"]
        if not (run / "formal_audit.json").exists():
            missing_training.append(spec["label"])
            continue
        actual = train_audit(run, spec, protocol) if task == "cube" else train_audit(run)
        require(actual == read(run / "formal_audit.json"), "Stored training audit differs: " + spec["label"])
        entry = dict(audit_sha256=digest(run / "formal_audit.json"))
        if task == "cube":
            entry["contact_disposition"] = contact_disposition(run)
        training[spec["label"]] = entry
        print("RECHECKED_TRAINING", task, spec["label"], flush=True)
    directories, missing_evals = collect_reports(roots, jobs)
    expected = {job["label"]: job for job in jobs}
    declared = set()
    for root in roots:
        if not (root / "manifest.json").exists():
            require(not any(root.glob("*/report.json")), "Reports without an evaluator manifest")
            continue
        labels = check_manifest(root, suite, expected)
        require(not (labels & declared), "Overlapping evaluation shards")
        declared.update(labels)
        require(all(p.parent.name in labels for p in root.glob("*/report.json")), "Undeclared report in shard")
    evaluations, nominal = {}, {}
    nominal_runs = {r["label"] for r in specifications if task != "cube" or r["phase"] == "main"}
    for label, directory in directories.items():
        actual = eval_audit(directory, expected[label], suite)
        require(actual == read(directory / "independent_audit.json"), "Stored evaluation audit differs: " + label)
        evaluations[label] = dict(report_sha256=digest(directory / "report.json"),
                                  audit_sha256=digest(directory / "independent_audit.json"))
        if label.endswith("__nominal") and label.removesuffix("__nominal") in nominal_runs:
            nominal[label.removesuffix("__nominal")] = read(directory / "report.json")
        print("RECHECKED_EVALUATION", task, label, flush=True)
    statistics = analyze(suite, roots, output / task / "statistics", allow_partial=True)
    statistics_match = False
    if not missing_training and not missing_evals:
        require(declared == expected.keys(), "Incomplete evaluator manifest coverage")
        require(json.loads(json.dumps(statistics)) == read(suite / "report/statistics.json"),
                "Published statistics differ from recomputation")
        statistics_match = True
    videos = dict(passed=False, reason="Formal nominal evaluations or video manifest not complete")
    nominal_count = 40 if task == "cube" else 20
    if len(nominal) == nominal_count and (video_root / "manifest.json").exists():
        videos = check_videos(video_root, nominal, task)
    return dict(training=training, missing_training=missing_training, evaluations=evaluations,
                missing_evaluations=missing_evals, statistics_recomputed=True,
                published_statistics_match=statistics_match, videos=videos,
                machine_checks_passed=not missing_training and not missing_evals and statistics_match and videos["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for task in ("cube", "cap"):
        parser.add_argument("--" + task + "-suite", type=Path, required=True)
        parser.add_argument("--" + task + "-evaluations", type=Path, nargs="+", required=True)
        parser.add_argument("--" + task + "-videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(start_unix=time.time(), verifier_sha256=digest(Path(__file__)), tasks={}, errors={},
                  entire_goal_complete=False,
                  manual_requirements=["Inspect palm/side-view success and failure videos, not only file metadata",
                                       "Reconcile final Chinese report with actual statistics and failure evidence",
                                       "Verify all user requirements including object alignment and deployment limitations"])
    for task in ("cube", "cap"):
        try:
            result["tasks"][task] = audit_task(task, getattr(args, task + "_suite"),
                getattr(args, task + "_evaluations"), getattr(args, task + "_videos"), args.output)
        except Exception as error:
            result["errors"][task] = dict(type=type(error).__name__, message=str(error))
        (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    result["machine_checks_passed"] = (not result["errors"] and len(result["tasks"]) == 2
                                      and all(t["machine_checks_passed"] for t in result["tasks"].values()))
    result["end_unix"] = time.time()
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(machine_checks_passed=result["machine_checks_passed"], errors=result["errors"])), flush=True)
    raise SystemExit(0 if result["machine_checks_passed"] else 2)


if __name__ == "__main__":
    main()
