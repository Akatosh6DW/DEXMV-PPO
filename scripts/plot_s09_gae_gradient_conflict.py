"""Visualize existing matched-target diagnostics without pooling new samples."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from retarget_s09_reference import digest


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"artifacts/orca_benchmark_suite/S09/bc_repair/ppo_improvement_20261018/D41_gae_gradient_conflict"
    summary=json.loads((folder/"summary.json").read_text())
    assert summary["status"]=="completed_diagnostic_only"
    with (folder/"pairs.csv").open() as stream:
        pairs=list(csv.DictReader(stream))
    with (folder/"target_comparison.csv").open() as stream:
        comparisons=list(csv.DictReader(stream))
    within=np.full((6,2),np.nan)
    global_=np.full((6,2),np.nan)
    for row in pairs:
        if row["a_phase"]=="acquisition" and row["b_phase"]=="released_before_30s" and row["a_mode"]==row["b_mode"]!="-1":
            within[int(row["a_mode"]),["mc","gae20"].index(row["target"])]=float(row["cosine"])
    for row in comparisons:
        if row["phase"]=="acquisition" and row["mode"]!="-1":
            global_[int(row["mode"])]=[float(row["mc_global_cosine"]),float(row["gae_global_cosine"])]
    assert np.isfinite(within).all() and np.isfinite(global_).all()
    fig,axes=plt.subplots(1,2,figsize=(8,4.7),constrained_layout=True)
    for ax,values,title in zip(axes,(within,global_),
        ("Acquisition vs holding\nwithin each pose bin","Acquisition vs global\nvalue-loss gradient")):
        im=ax.imshow(values,vmin=-1,vmax=1,cmap="RdBu",aspect="auto")
        ax.set(title=title,xticks=[0,1],xticklabels=["MC","GAE-20"],yticks=np.arange(6),ylabel="Pose bin")
        for i in range(6):
            for j in range(2):
                ax.text(j,i,f"{values[i,j]:+.2f}",ha="center",va="center",
                        color="white" if abs(values[i,j])>.65 else "black")
    fig.colorbar(im,ax=axes,label="Gradient cosine (negative = conflicting directions)",shrink=.8)
    fig.suptitle("D41: fixed-critic diagnostic, 32 rollouts; not a success-rate result",fontsize=11)
    path=folder/"gradient_targets.png"
    fig.savefig(path,dpi=180)
    plt.close(fig)
    record=dict(source_sha256=digest(Path(__file__)),summary_sha256=digest(folder/"summary.json"),
                pairs_sha256=digest(folder/"pairs.csv"),comparison_sha256=digest(folder/"target_comparison.csv"),
                image_sha256=digest(path))
    (folder/"plot_provenance.json").write_text(json.dumps(record,indent=2)+"\n")
    print(path)


if __name__=="__main__":
    main()
