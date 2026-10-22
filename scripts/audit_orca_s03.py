"""Independent S03 coverage, identities, scoring and paired-reset audit."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_orca_suite import check_preview
from run_orca_s03_confirmation import ROOT, OUT, SEEDS, evaluation_jobs, training_jobs, digest
from run_orca_s03_development import BASE, candidate, rank


def verify_trace(trace, episodes, config, force):
    assert all(np.isfinite(v).all() for v in trace.values())
    expected = np.zeros(800)
    expected[200:204] = force
    np.testing.assert_allclose(np.linalg.norm(trace["push"], axis=-1),
                               np.broadcast_to(expected, trace["push"].shape[:2]), atol=1e-6)
    # Reuse the frozen, independent contact/hold/action recomputation, and check
    # S03's variable force separately instead of weakening the old B5 assertion.
    neutral = dict(trace, push=np.zeros_like(trace["push"]))
    check_preview(neutral, episodes, dict(config, benchmark="B0_nominal",
                  residual_gain=0. if config["method"] == "zero" else 1.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-videos", action="store_true")
    args = parser.parse_args()
    development = json.loads((OUT / "development_protocol.json").read_text())
    protocol = json.loads((OUT / "confirmation_protocol.json").read_text())
    eval_plan = json.loads((OUT / "evaluation_protocol.json").read_text())
    selected = json.loads((OUT / "selected_recipes.json").read_text())
    force = json.loads((OUT / "force_selection.json").read_text())["selected_force_n"]
    assert protocol["selected_recipes_sha256"] == digest(OUT / "selected_recipes.json")
    assert protocol["development_sha256"] == digest(OUT / "development_protocol.json")
    assert eval_plan["confirmation_protocol_sha256"] == digest(OUT / "confirmation_protocol.json")
    assert eval_plan["force_selection_sha256"] == digest(OUT / "force_selection.json")
    assert protocol["training_jobs"] == list(training_jobs(selected["recipes"]))
    assert eval_plan["jobs"] == list(evaluation_jobs(selected["recipes"], force))
    for frozen in (development["source_hashes"], protocol["source_hashes"]):
        for path, expected in frozen.items():
            assert digest(ROOT / path) == expected, path
    incumbents, scores = dict(BASE), {m: None for m in BASE}
    for index in range(4):
        decision = json.loads((OUT / "development_rounds" / f"r{index}.json").read_text())
        for method in BASE:
            entry = decision["decisions"][method]
            recipe = candidate(method, index, incumbents[method])
            assert entry["candidate"] == recipe
            trials = []
            for seed in (20001, 20002):
                path = OUT / "development" / method / f"r{index}_s{seed}" / "summary.json"
                trials.append(json.loads(path.read_text()))
            assert sorted(trials, key=lambda s: s["job_sha256"]) == entry["trial_summaries"]
            score = rank(trials)
            accepted = scores[method] is None or score > scores[method]
            assert entry["score"] == list(score) and entry["accepted"] == accepted
            if accepted:
                incumbents[method], scores[method] = recipe, score
            assert entry["retained_recipe"] == incumbents[method]
    assert incumbents == selected["recipes"]
    missing, training_checked, starts = [], 0, {}
    episode_count, preview_count, final_jobs_checked = 0, 0, 0
    files = sorted((OUT / "jobs").glob("development__*.json"))
    files += sorted((OUT / "jobs").glob("confirmation__*.json"))
    assert len(files) == 64
    configs = [json.loads(path.read_text()) for path in files] + eval_plan["jobs"]
    for job in configs:
        name = job["output"].split("/S03/")[1].replace("/", "__")
        path = OUT / "jobs" / (name+".json")
        run = ROOT / job["output"]
        assert json.loads(path.read_text()) == job
        if not (run / "summary.json").exists():
            missing.append(job["output"])
            continue
        summary = json.loads((run / "summary.json").read_text())
        config = json.loads((run / "run_config.json").read_text())
        assert summary["status"] == "completed"
        assert summary["job_sha256"] == config["job_sha256"] == digest(path)
        for key, value in job.items():
            assert config[key] == value, (name, key)
        for filename, expected in config["source_sha256"].items():
            assert digest(run / "sources" / filename) == expected
            if "scripts/"+filename in protocol["source_hashes"]:
                assert protocol["source_hashes"]["scripts/"+filename] == expected
        for key in ("model", "reference"):
            assert digest(Path(config[key])) == config[key+"_sha256"]
        if job.get("steps", 0):
            with (run / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            steps = [int(r["step"]) for r in rows]
            expected_steps = list(range(0, job["steps"]+1, 81920))
            assert steps == expected_steps
            values = [float(r["clean_success30"]) for r in rows]
            assert summary["first"] == values[0] and summary["last"] == values[-1]
            assert summary["max"] == max(values)
            assert summary["best_step"] == steps[values.index(max(values))]
            assert summary["final_step"] == job["steps"]
            np.testing.assert_allclose(summary["clean_auc"], np.trapezoid(values, steps)/steps[-1])
            for step, row in zip(steps, rows):
                directory = run / f"step_{step:09d}"
                assert (directory / "params.pkl").stat().st_size > 1000
                episodes = json.loads((directory / "episodes.json").read_text())
                assert [e["seed"] for e in episodes] == list(range(61000, 61016))
                np.testing.assert_allclose(np.mean([e["clean_success30"] for e in episodes]), float(row["clean_success30"]))
                with np.load(directory / "preview_trajectories.npz") as z:
                    verify_trace(dict(z), episodes[:4], config, 0.)
            assert digest(run / "params.pkl") == digest(run / f"step_{steps[-1]:09d}" / "params.pkl")
            training_checked += 1
            continue
        if job.get("checkpoint"):
            assert config["checkpoint_sha256"] == digest(ROOT / job["checkpoint"])
        modes = ("nominal_closed", "nominal_replay", "pushed_closed", "pushed_replay") if job.get("paired_replay") else ("evaluation",)
        mechanism_traces = {}
        for mode in modes:
            directory = run / mode
            episodes = json.loads((directory / "episodes.json").read_text())
            assert [e["seed"] for e in episodes] == list(range(job["eval_start"], job["eval_start"]+200))
            row = summary["summaries"][mode]
            for key in ("success30", "clean_success30"):
                assert all(isinstance(e[key], bool) for e in episodes)
                np.testing.assert_allclose(row[key], np.mean([e[key] for e in episodes]))
            for e in episodes:
                assert e["clean_success30"] == (e["success30"] and not e["substep_forbidden"] and e["post_release_penetration_m"] <= .003)
                assert e["full_episode_clean_success30"] == (e["clean_success30"] and e["max_penetration_m"] <= .003)
            with np.load(directory / "initial_states.npz") as z:
                np.testing.assert_array_equal(z["seeds"], [e["seed"] for e in episodes])
                qpos = z["qpos"].copy()
            assert qpos.shape == (200, 24) and np.isfinite(qpos).all()
            condition = job["condition"]
            if condition in starts:
                np.testing.assert_array_equal(qpos, starts[condition])
            else:
                starts[condition] = qpos
            with np.load(directory / "preview_trajectories.npz") as z:
                trace = dict(z)
            assert trace["qpos"].shape == (4, 800, 24)
            force_n = job.get("push_force", .15 if job.get("benchmark") == "B5_push" else 0.)
            if mode.startswith("nominal"):
                force_n = 0.
            verify_trace(trace, episodes[:4], config, force_n)
            if job.get("paired_replay"):
                mechanism_traces[mode] = trace
            episode_count += 200
            preview_count += 4
        if job.get("paired_replay"):
            assert all(config["replay_audit"][k] for k in ("nominal_active_trajectory_exact", "pre_push_active_trajectory_exact", "replay_actions_exact"))
            with np.load(run / "nominal_recorded_actions.npz") as z:
                actions, active = z["actions"], z["active"]
                assert actions.shape == (200, 800, 17)
                assert np.all(actions[~active] == 0)
            nominal = mechanism_traces["nominal_closed"]
            np.testing.assert_array_equal(actions[:4], np.where(nominal["active"][..., None], nominal["policy_action"], 0))
            for mode, trace in mechanism_traces.items():
                count = 800 if mode.startswith("nominal") else 200
                mask = nominal["active"][:, :count]
                np.testing.assert_array_equal(nominal["qpos"][:, :count][mask], trace["qpos"][:, :count][mask])
                if mode.endswith("replay"):
                    np.testing.assert_array_equal(trace["policy_action"], actions[:4])
        final_jobs_checked += 1
    if not missing:
        assert training_checked == 64 and final_jobs_checked == 738
        assert episode_count == 153600 and preview_count == 3072 and len(starts) == 9
    videos_checked = 0
    if args.require_videos:
        manifest = json.loads((OUT / "videos/manifest.json").read_text())
        assert manifest["complete"] and len(manifest["cases"]) == 50
        for case in manifest["cases"]:
            assert case["train_seed"] in (None, 21001)
            assert case["test_seed"] in (71000, 72000, 73000)
            for video in case["videos"].values():
                assert digest(OUT / video["path"]) == video["sha256"]
                videos_checked += 1
        assert videos_checked == 100
        stats = json.loads((OUT / "report/statistics.json").read_text())
        assert stats["complete"] and stats["episodes"] == episode_count
        assert len([c for c in stats["comparisons"] if c["primary"]]) == 3
    result = dict(complete=not missing, missing=missing, training_checked=training_checked,
                  final_jobs_checked=final_jobs_checked, final_episodes=episode_count,
                  independently_recomputed_preview_episodes=preview_count, videos_checked=videos_checked,
                  scope="All identities, episode score algebra and paired reset states. Independent time-series recomputation covers four saved cases per evaluation and checkpoint, not every episode.",
                  source_sha256=digest(Path(__file__)), confirmation_protocol_sha256=digest(OUT / "confirmation_protocol.json"),
                  evaluation_protocol_sha256=digest(OUT / "evaluation_protocol.json"))
    (OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "missing"}))
    if missing and not args.allow_partial:
        raise RuntimeError("Incomplete matrix")


if __name__ == "__main__":
    main()
