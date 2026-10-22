"""Descriptive cube/cylinder diagnostics, explicitly not a shape-only ablation."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from run_s07_study import OUT, write_json
from prepare_cylinder_s02_source import digest


def main():
    target = OUT / "cross_task_diagnosis"
    target.mkdir(exist_ok=True)
    sources = [("Cube S05 (4 dev states)", OUT.parent / "S05/T10_pad_guided_long", OUT / "bc_cube_diagnosis", OUT / "cube_bc_noise_std01"),
               ("Cylinder S07 (8 dev states)", OUT / "train/guided_s7102", OUT / "bc_diagnosis_7102", OUT / "bc_noise_std01")]
    summaries, metrics_rows = [], []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for name, run, diagnosis, noisy in sources:
        config = json.loads((run / "run_config.json").read_text())
        m = mujoco.MjModel.from_xml_path(config["model"])
        with (run / "metrics.csv").open() as f:
            metrics = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        with (diagnosis / "checkpoint_drift.csv").open() as f:
            drift = list(csv.DictReader(f))
        training = [json.loads(line) for line in (run / "ppo_metrics.jsonl").read_text().splitlines()]
        noise = json.loads((noisy / "summary.json").read_text())["rows"][0]
        with np.load(noisy / "step_000000000/initial_states.npz") as a, np.load(run / "step_000000000/initial_states.npz") as b:
            np.testing.assert_array_equal(a["qpos"], b["qpos"])
        assert json.loads((noisy / "run_config.json").read_text())["checkpoint_sha256"] == digest(run / "step_000000000/params.pkl")
        with np.load(config["dataset"]) as z:
            dataset_shape = z["obs"].shape
        summaries.append(dict(task=name, run=str(run), model_sha256=digest(config["model"]),
            seed=config["seed"], bc_updates=config["bc_updates"], bc_episodes=dataset_shape[0],
            bc_observations=dataset_shape[0]*dataset_shape[1], steps=config["steps"], std=config["std"],
            learning_rate=config["learning_rate"], evals=config["evals"], eval_count=config["eval_count"],
            mass=float(m.body("cube").mass[0]), geom_type=int(m.geom("cube").type[0]),
            friction=m.geom("cube").friction.tolist(), impratio=float(m.opt.impratio), integrator=int(m.opt.integrator),
            initial_development=metrics[0], final_development=metrics[-1],
            first_target_error_rad=float(drift[0]["target_rms_rad"]), last_target_error_rad=float(drift[-1]["target_rms_rad"]),
            log_kl_first=training[0]["training/kl_mean"], log_kl_max=max(r["training/kl_mean"] for r in training),
            noisy_frozen_bc=noise, noisy_initial_states_match=True))
        metrics_rows.extend(dict(task=name, **r) for r in metrics)
        axes[0,0].plot([r["step"] for r in metrics], [r["success30"]*100 for r in metrics], "o-", label=name)
        axes[0,1].plot([int(r["step"]) for r in drift], [float(r["target_rms_rad"]) for r in drift], "o-", label=name)
        axes[1,0].semilogy([r["step"] for r in training], [r["training/kl_mean"] for r in training], "o-", label=name)
    axes[0,0].set(title="Development success, not matched shape test", ylabel="Strict success (%)", xlabel="Online transitions", ylim=(-3,103))
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(title="Both move away from demonstration labels", ylabel="Fixed-validation target RMS (rad)", xlabel="Online transitions")
    axes[1,0].set(title="Logged old/new distribution KL", ylabel="KL (log scale)", xlabel="Online transitions")
    axes[1,1].bar(["Cube, n=4", "Cylinder, n=8"], [r["noisy_frozen_bc"]["broad_hold_sec"] for r in summaries], color=["#4486b8", "#b66a3c"])
    axes[1,1].set(title="Frozen BC + std=0.1: both strict success=0", ylabel="Mean broad hold (s)", ylim=(0,40))
    fig.savefig(target / "cube_cylinder_diagnosis.png", dpi=160)
    plt.close(fig)
    with (target / "development_curves.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0]))
        w.writeheader()
        w.writerows(metrics_rows)
    write_json(target / "comparison.json", dict(tasks=summaries,
        limitation="Descriptive comparison of existing tasks: different reference, BC data/budget, seed, contact criterion, solver and evaluation count. Cannot attribute effects solely to object shape."))
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
