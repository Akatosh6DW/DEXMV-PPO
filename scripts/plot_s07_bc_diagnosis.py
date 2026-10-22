"""Plot fixed-state drift and paired, no-update BC exploration probes."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_s07_study import OUT, write_json
from prepare_cylinder_s02_source import digest


def main():
    target = OUT / "bc_diagnosis_7102"
    with (target / "checkpoint_drift.csv").open() as f:
        drift = list(csv.DictReader(f))
    original = OUT / "train/guided_s7102/step_000000000"
    checkpoint_hash = digest(original / "params.pkl")
    source = [("Deterministic BC", original, None),
              ("Sample std=0.1", OUT / "bc_noise_std01/step_000000000", .1),
              ("Sample std=0.01", OUT / "bc_noise_std001/step_000000000", .01)]
    episodes, summaries, initial = [], [], None
    for label, directory, std in source:
        with np.load(directory / "initial_states.npz") as z:
            if initial is None:
                initial = z["qpos"].copy()
            else:
                np.testing.assert_array_equal(initial, z["qpos"])
        if std is not None:
            config = json.loads((directory.parent / "run_config.json").read_text())
            probe = json.loads((directory.parent / "noise_probe.json").read_text())
            assert config["checkpoint_sha256"] == checkpoint_hash
            assert config["steps"] == 0 and probe["optimizer_updates"] == 0
            assert probe["probe_std"] == std
        records = json.loads((directory / "episodes.json").read_text())
        episodes.extend(dict(mode=label, **r) for r in records)
        summaries.append(dict(mode=label, successes=sum(r["success30"] for r in records), count=len(records),
                              mean_strict_hold=float(np.mean([r["pad_hold_sec"] for r in records])),
                              mean_broad_hold=float(np.mean([r["broad_hold_sec"] for r in records])),
                              max_depth_mm=max(r["penetration_m"] for r in records)*1000,
                              max_force_n=max(r["peak_force_n"] for r in records)))
    with (target / "noise_episodes.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0]))
        w.writeheader()
        w.writerows(episodes)
    write_json(target / "noise_comparison.json", dict(checkpoint_sha256=checkpoint_hash,
        paired_initial_states=True, optimizer_updates=0, methods=summaries,
        limitation="One BC seed, eight development initial states and one noise realization per state; not independent test or retraining ablation"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    steps = [int(r["step"]) for r in drift]
    axes[0].plot(steps, [float(r["target_rms_rad"]) for r in drift], "o-", label="Error to demonstration")
    axes[0].plot(steps, [float(r["bc_mean_drift_max_rad"]) for r in drift], "o-", label="Max drift from BC")
    axes[0].set(xlabel="PPO transitions", ylabel="Joint target radians", title="Same 2400 BC validation observations")
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, [float(r["fixed_state_kl_mean"]) for r in drift], "o-", color="#258b78")
    axes[1].set(xlabel="PPO transitions", ylabel="Mean KL from initial BC", title="Fixed states, not on-policy log KL")
    rates = [r["successes"]/r["count"]*100 for r in summaries]
    axes[2].bar(range(3), rates, color=["#4486b8", "#b66a3c", "#258b78"])
    for i, r in enumerate(summaries):
        axes[2].text(i, rates[i]+2, f"{r['successes']}/{r['count']}", ha="center")
    axes[2].set(xticks=range(3), xticklabels=["No noise", "std=0.1", "std=0.01"], ylim=(0, 103),
                ylabel="Strict success (%)", title="Same frozen BC, zero learning updates")
    fig.savefig(target / "bc_diagnosis.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
