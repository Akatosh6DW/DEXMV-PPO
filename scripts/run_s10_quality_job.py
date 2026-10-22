"""Add acquisition-path diagnostics to S10 evaluation without changing physics."""

import numpy as np

import run_s10_job as runner

original_summarize=runner.summarize


def quality_summary(trace,seeds):
    summary,episodes=original_summarize(trace,seeds)
    for i,row in enumerate(episodes):
        active=trace["active"][i].astype(bool)
        audit=trace["audit"][i]
        released=active&(audit[:,10]>0)
        acquiring=active&~released
        root=trace["qpos"][i,active,:3]
        target=trace["mocap"][i,active,1,:]
        row.update(max_path_penetration_m=float(np.max(audit[active,6],initial=0)),
            peak_acquisition_force_n=float(np.max(audit[acquiring,7],initial=0)),
            peak_released_force_n=float(np.max(audit[released,7],initial=0)),
            max_tracking_error_m=float(np.linalg.norm(root-target,axis=1).max()),
            hand_travel_m=float(np.linalg.norm(np.diff(root,axis=0),axis=1).sum()))
        row["full_path_success30"]=bool(row["success30"] and row["max_path_penetration_m"]<=.003)
    summary.update(full_path_success30=float(np.mean([r["full_path_success30"] for r in episodes])),
                   max_path_penetration_m=max(r["max_path_penetration_m"] for r in episodes),
                   max_tracking_error_m=max(r["max_tracking_error_m"] for r in episodes))
    return summary,episodes


if __name__=="__main__":
    runner.summarize=quality_summary
    runner.main()
