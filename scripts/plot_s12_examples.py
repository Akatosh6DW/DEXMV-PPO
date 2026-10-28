"""Plot the same preselected held-out state for every controller, including failures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from analyze_s12_study import COLORS, LABELS, ORDER, csv_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.study/"protocol.json").read_text())
    cases = json.loads((a.study/"evaluation_plan.json").read_text())["jobs"]
    seed, source = plan["conditions"]["nominal"]["seeds"][0],plan["train_seeds"][0]
    chosen = {j["label"]:j for j in cases if j["condition"]=="nominal" and j["train_seed"] in (None,source)}
    if set(chosen)!=set(ORDER):
        raise ValueError("Missing planned example controllers")
    fig,axes = plt.subplots(3,2,figsize=(12,10))
    examples = []
    for method in ORDER:
        run = Path(chosen[method]["output"])
        file = run/f"trajectory_{seed}.npz"
        with np.load(file) as z:
            t = z["time"]
            values = (z["vertices"].mean(1)[:,2], z["hold_sec"], 100*z["max_principal_strain"],
                      1000*z["penetration_m"], z["forbidden_force_n"],z["peak_force_n"])
            for ax,value in zip(axes.flat,values):
                ax.plot(t,value,label=LABELS[method],color=COLORS[method],linewidth=1.3)
            examples.append(dict(method=method,train_seed=chosen[method]["train_seed"],initial_seed=seed,
                trace=str(file),success15=bool(z["success15"][-1]),duration_sec=float(t[-1])))
    labels = ("Object center height (m)","Current strict hold (s)","Maximum principal strain (%)",
              "Maximum penetration (mm)","Forbidden contact load (N)","Largest single contact load (N)")
    for ax,label in zip(axes.flat,labels):
        ax.set(xlabel="Time (s)",ylabel=label,xlim=(0,24))
        ax.axvspan(5,5.5,color="gray",alpha=.12)
        ax.grid(alpha=.2)
    axes[0,0].axhline(.535,linestyle="--",color="gray",linewidth=1)
    axes[0,1].axhline(15,linestyle="--",color="gray",linewidth=1)
    axes[1,0].axhline(30,linestyle="--",color="gray",linewidth=1)
    axes[1,1].axhline(3,linestyle="--",color="gray",linewidth=1)
    axes[2,0].axhline(.02,linestyle="--",color="gray",linewidth=1)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f"Same preselected state {seed}, first training seed {source}; not an average success rate")
    fig.tight_layout()
    output = a.study/"report"
    fig.savefig(output/"paired_example.png",dpi=160)
    plt.close(fig)
    (output/"paired_example.json").write_text(json.dumps(examples,indent=2)+"\n")
    fig,axes = plt.subplots(2,2,figsize=(11,7))
    for method in ("reference","bc_initial","guided","residual"):
        file = Path(chosen[method]["output"])/f"trajectory_{seed}.npz"
        with np.load(file) as z:
            for ax,column in zip(axes.flat,(8,9,11,15)):
                ax.plot(z["time"],z["ctrl"][:,column],label=LABELS[method],color=COLORS[method])
    for ax,label in zip(axes.flat,("Middle MCP","Middle PIP","Index MCP","Thumb MCP")):
        ax.set(title=label,xlabel="Time (s)",ylabel="Executed target (rad)",xlim=(0,10))
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Recorded actuator targets, including clipping and rate limiting")
    fig.tight_layout();fig.savefig(output/"example_targets.png",dpi=160);plt.close(fig)
    reference_run = Path(chosen["reference"]["output"])
    model = mujoco.MjModel.from_xml_path(str(reference_run/"model.xml"))
    comparisons = []
    with np.load(reference_run/f"trajectory_{seed}.npz") as reference:
        for job in cases:
            if job["condition"] != "nominal" or job["label"] != "bc_initial":
                continue
            with np.load(Path(job["output"])/f"trajectory_{seed}.npz") as bc:
                np.testing.assert_array_equal(reference["time"],bc["time"])
                error = bc["ctrl"]-reference["ctrl"]
                for column in range(17):
                    peak = int(np.argmax(abs(error[:,column])))
                    comparisons.append(dict(train_seed=job["train_seed"],initial_seed=seed,
                        actuator=model.actuator(column).name,
                        executed_target_rms_difference_rad=float(np.sqrt(np.mean(error[:,column]**2))),
                        peak_absolute_difference_rad=float(abs(error[peak,column])),
                        peak_time_sec=float(reference["time"][peak]),
                        after_lift_rms_difference_rad=float(np.sqrt(np.mean(
                            error[reference["time"]>=8,column]**2)))))
    csv_write(output/"bc_reference_target_difference.csv",comparisons)
    print(output)


if __name__=="__main__":
    main()
