"""Evaluate unchanged policies with full-case contact diagnostics.

The frozen S09 success criterion remains primary. Whole-episode penetration
is an additional diagnostic, never a replacement for the primary result.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

import run_s09_ppo_improvement as improvement
from retarget_s09_reference import digest


def quality_episodes(trace, episodes):
    output=[]
    for i,episode in enumerate(episodes):
        active=trace["active"][i].astype(bool)
        audit=trace["audit"][i]
        released=active & (audit[:,10]>0)
        before=active & ~released
        valid=released & (audit[:,11]>0)
        breaks=np.r_[False,valid[:-1]] & released & ~valid
        all_depth=float(np.max(audit[active,6],initial=0))
        paired=np.minimum(audit[:,0],np.max(audit[:,1:5],axis=-1))
        effective=trace["effective_residual"][i,active]
        output.append(dict(episode,
            effective_target_residual_rms_rad=float(np.sqrt(np.mean(effective**2))),
            max_effective_target_residual_rad=float(np.max(np.abs(effective),initial=0)),
            valid_streak_break_events=int(np.sum(breaks)),
            total_valid_grasp_sec=float(np.sum(valid)*.05),
            max_all_penetration_m=all_depth,
            max_pre_release_penetration_m=float(np.max(audit[before,6],initial=0)),
            peak_pre_release_force_n=float(np.max(audit[before,7],initial=0)),
            peak_released_force_n=float(np.max(audit[released,7],initial=0)),
            p95_released_control_peak_force_n=float(np.quantile(audit[released,7],.95)) if released.any() else None,
            pre_release_forbidden_contact=bool(np.any(audit[before,5]>0)),
            valid_contact_load_below_0_2n_fraction=float(np.mean(paired[valid]<.2)) if valid.any() else None,
            success30_all_depth_le_3mm=bool(episode["success30"] and all_depth<=.003)))
    return output


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--steps",type=int,required=True,choices=(0,))
    parser.parse_known_args()
    output=Path(sys.argv[sys.argv.index("--output")+1])
    runner=improvement.runner
    original=runner.summarize

    def summarize(trace,seeds):
        row,episodes=original(trace,seeds)
        episodes=quality_episodes(trace,episodes)
        row=dict(row,
            effective_target_residual_rms_rad=float(np.mean([e["effective_target_residual_rms_rad"] for e in episodes])),
            max_all_penetration_m=max(e["max_all_penetration_m"] for e in episodes),
            max_pre_release_penetration_m=max(e["max_pre_release_penetration_m"] for e in episodes),
            peak_pre_release_force_n=max(e["peak_pre_release_force_n"] for e in episodes),
            peak_released_force_n=max(e["peak_released_force_n"] for e in episodes),
            success30_all_depth_le_3mm=float(np.mean([e["success30_all_depth_le_3mm"] for e in episodes])))
        return row,episodes

    runner.summarize=summarize
    try:
        improvement.main()
    finally:
        runner.summarize=original
        if (output/"sources").exists():
            config_path=output/"improvement_config.json"
            run_config=json.loads((output/"run_config.json").read_text())
            if run_config["method"] == "reference" and config_path.exists():
                config=json.loads(config_path.read_text())
                config["action_architecture"]="fixed reference replay without learned correction"
                config_path.write_text(json.dumps(config,indent=2)+"\n")
            metadata=dict(primary_success_unchanged=True,scope="All evaluated cases, active steps only",
                extra_depth_criterion_is_secondary=True,
                effective_target_residual_definition="RMS of executed control target minus reference base over active steps and actuators; includes joint-limit clipping, not actual joint displacement. Zero by convention for full-action guided policies.",
                force_quantile_definition="95th percentile of per-control-step maximum contact force",
                low_load_definition="Among valid released steps, min(thumb load, maximum opposing load) < 0.2 N",
                streak_break_definition="A previously valid released grasp becomes invalid at the next active control step; not necessarily a drop",
                source_sha256=digest(output/"sources"/Path(__file__).name))
            (output/"quality_config.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__":
    main()
