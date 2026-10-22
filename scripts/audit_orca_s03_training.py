"""Audit completed S03 training jobs while the unchanged queue is running."""

import csv
import json
from pathlib import Path
import time

import numpy as np

from audit_orca_s03 import verify_trace
from run_orca_s03_confirmation import OUT, ROOT, digest


def main():
    protocol = json.loads((OUT / "confirmation_protocol.json").read_text())
    for path, expected in protocol["source_hashes"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("Frozen input changed: " + path)
    initial = None
    checked, pending = [], []
    for job in protocol["training_jobs"]:
        run = ROOT / job["output"]
        if not (run / "summary.json").exists():
            pending.append(job["output"])
            continue
        name = job["output"].split("/S03/")[1].replace("/", "__")
        path = OUT / "jobs" / (name+".json")
        assert json.loads(path.read_text()) == job
        summary = json.loads((run / "summary.json").read_text())
        config = json.loads((run / "run_config.json").read_text())
        assert summary["job_sha256"] == config["job_sha256"] == digest(path)
        assert summary["status"] == "completed"
        for key, value in job.items():
            assert config[key] == value
        for filename, expected in config["source_sha256"].items():
            assert digest(run / "sources" / filename) == expected
            if "scripts/"+filename in protocol["source_hashes"]:
                assert expected == protocol["source_hashes"]["scripts/"+filename]
        with (run / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        steps = [int(row["step"]) for row in rows]
        values = [float(row["clean_success30"]) for row in rows]
        assert steps == list(range(0, 655361, 81920))
        assert summary["first"] == values[0] and summary["last"] == values[-1]
        assert summary["max"] == max(values) and summary["best_step"] == steps[values.index(max(values))]
        assert summary["final_step"] == 655360
        np.testing.assert_allclose(summary["clean_auc"], np.trapezoid(values, steps)/655360)
        for row, step in zip(rows, steps):
            directory = run / f"step_{step:09d}"
            assert (directory / "params.pkl").stat().st_size > 1000
            episodes = json.loads((directory / "episodes.json").read_text())
            assert [e["seed"] for e in episodes] == list(range(61000, 61016))
            for score in ("success30", "clean_success30"):
                assert all(isinstance(e[score], bool) for e in episodes)
                np.testing.assert_allclose(float(row[score]), np.mean([e[score] for e in episodes]))
            with np.load(directory / "initial_states.npz") as z:
                np.testing.assert_array_equal(z["seeds"], list(range(61000, 61016)))
                if initial is None:
                    initial = z["qpos"].copy()
                np.testing.assert_array_equal(z["qpos"], initial)
            with np.load(directory / "preview_trajectories.npz") as z:
                verify_trace(dict(z), episodes[:4], config, 0.)
        assert digest(run / "params.pkl") == digest(run / "step_000655360/params.pkl")
        checked.append(dict(job=job["output"], job_sha256=digest(path),
                            checkpoints=len(rows), independent_preview_cases=4*len(rows),
                            first=summary["first"], last=summary["last"], max=summary["max"],
                            best_step=summary["best_step"], raw_last=summary["raw_last"]))
    repeated_recipe_checks = []
    for seed in range(21001, 21011):
        fixed = OUT / "confirmation/ppo_fixed" / f"s{seed}"
        tuned = OUT / "confirmation/ppo_tuned" / f"s{seed}"
        if not all((run / "summary.json").exists() for run in (fixed, tuned)):
            continue
        c1, c2 = [json.loads((run / "run_config.json").read_text()) for run in (fixed, tuned)]
        if any(c1[key] != c2[key] for key in ("std", "learning_rate", "shaping")):
            continue
        matches = [digest(fixed / f"step_{step:09d}/params.pkl") == digest(tuned / f"step_{step:09d}/params.pkl")
                   for step in range(0, 655361, 81920)]
        # A mismatch is a reproducibility finding, not permission to discard runs.
        repeated_recipe_checks.append(dict(seed=seed, checkpoint_weights_bitwise_equal=matches,
                                           all_equal=all(matches)))
    result = dict(epoch_time=time.time(), confirmation_complete=not pending,
                  checked_jobs=len(checked), pending_jobs=len(pending), checked=checked,
                  independently_recomputed_preview_cases=sum(r["independent_preview_cases"] for r in checked),
                  repeated_recipe_checks=repeated_recipe_checks,
                  paired_development_initial_states_exact=True if checked else None,
                  confirmation_protocol_sha256=digest(OUT / "confirmation_protocol.json"),
                  source_sha256=digest(Path(__file__)),
                  scope="Completed confirmation jobs only; nine checkpoints, all episode aggregates and initial states, four independently rescored traces per checkpoint. Not final held-out evaluation.")
    (OUT / "training_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("checked", "repeated_recipe_checks")}))


if __name__ == "__main__":
    main()
