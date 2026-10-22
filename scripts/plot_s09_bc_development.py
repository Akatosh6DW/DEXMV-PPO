"""Plot measured development results; never label them held-out performance."""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s09-bc-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/orca_benchmark_suite/S09/bc_repair"


def read(name):
    with (BASE/name/"metrics.csv").open() as f:
        return [{key:float(value) for key,value in row.items()} for row in csv.DictReader(f)]


def main():
    out = BASE/"report"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1,3,figsize=(15,4.6),layout="constrained")
    initial = [read("I11_mode_ppo")[0]["success30"], read("I23_phase_bc")[0]["success30"]]
    axes[0].bar([0,1], np.asarray(initial)*100, color=["#777777", "#3c8c70"])
    axes[0].set(xticks=[0,1], xticklabels=["Full-input BC", "Phase-curriculum BC"],
                title="BC rollout, seed 9101")
    for name,label,color in (("I24_phase_ppo","LR 3e-4", "#c15e57"),
                             ("I25_phase_small_lr","LR 3e-5", "#3c8c70")):
        rows=read(name)
        axes[1].plot([r["step"] for r in rows], [100*r["success30"] for r in rows],
                     marker="o",label=label,color=color)
    axes[1].set(title="Same BC origin, PPO learning rate", xlabel="Online transitions")
    axes[1].legend(fontsize=8)
    for name,color in (("I29_long_s13101", "#3c8c70"), ("I30_long_s13102", "#c15e57"),
                        ("I31_long_s13103", "#4a86b7")):
        rows=read(name)
        axes[2].plot([r["step"] for r in rows], [100*r["success30"] for r in rows],
                     marker="o",label=name.split("_s")[-1],color=color)
    axes[2].set(title="Three independent training seeds", xlabel="Online transitions")
    axes[2].legend(title="Training seed",fontsize=8)
    for ax in axes:
        ax.set(ylim=(0,100),ylabel="Development strict 30s success (%)")
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Development only: same 32 states; final markers are not best-checkpoint selection")
    fig.savefig(out/"repair_development.png", dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for name,label in (("I10_mode_bc40000","Full-input BC"), ("I23_phase_bc","Phase-curriculum BC")):
        records=json.loads((BASE/name/"bc_loss.json").read_text())
        for ax,key in zip(axes,("train_loss","validation_loss")):
            ax.plot([r["step"] for r in records], [r[key] for r in records],label=label)
            ax.set(yscale="log", xlabel="BC gradient updates", ylabel="Latent-action MSE", title=key)
            ax.legend()
    fig.suptitle("Lower offline imitation error does not guarantee better rollout")
    fig.savefig(out/"bc_offline_loss.png", dpi=160)
    plt.close(fig)
    print(json.dumps(dict(output=str(out), plots=["repair_development.png","bc_offline_loss.png"])))


if __name__ == "__main__":
    main()
