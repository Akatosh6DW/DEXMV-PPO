"""Show saved video targets, bounded corrections, and physical joint tracking."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--bound", type=float, required=True)
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    with np.load(args.trace) as packet:
        trace = {k: packet[k][args.episode] for k in packet.files}
    active = trace["active"].astype(bool)
    trace = {k: v[active] for k, v in trace.items()}
    np.testing.assert_allclose(trace["requested_residual"], np.clip(trace["action"], -1, 1) * args.bound, atol=1e-6)
    np.testing.assert_allclose(trace["desired_ctrl"], np.clip(trace["video_base"] + trace["requested_residual"],
                                                          *model.actuator_ctrlrange.T), atol=1e-6)
    np.testing.assert_allclose(trace["effective_residual"], trace["ctrl"] - trace["video_base"], atol=1e-6)
    assert np.abs(trace["effective_residual"]).max() <= args.bound + 1e-5
    time = (np.arange(len(trace["ctrl"])) + 1) * .05
    qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    cap = "degrees" in trace
    joints = (0, 11, 13, 14) if cap else (11, 12, 13, 14)
    names = (("Wrist", "Index MCP", "Thumb CMC", "Thumb abduction") if cap
             else ("Index MCP", "Index PIP", "Thumb CMC", "Thumb abduction"))
    released = np.flatnonzero(trace["released"] > 0) if cap else []
    event_s = (float(time[released[0]]) if len(released) else None) if cap else 3.
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    for axis, joint, name in zip(axes[:4], joints, names):
        axis.plot(time, trace["video_base"][:, joint], "--", color="#777777", label="Video reference")
        axis.plot(time, trace["ctrl"][:, joint], color="#286bb4", label="Applied target")
        axis.plot(time, trace["qpos"][:, qids[joint]], color="#218752", alpha=.85, label="Actual joint")
        axis.set_ylabel(name + " (rad)")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[4].plot(time, np.abs(trace["requested_residual"]).max(axis=1), color="#c56c22", label="Max requested correction")
    axes[4].plot(time, np.abs(trace["effective_residual"]).max(axis=1), "--", color="#286bb4", label="Max applied target correction")
    axes[4].axhline(args.bound, color="#777777", linestyle=":", label="Configured bound")
    axes[4].set_ylim(0, args.bound * 1.2)
    axes[4].set_ylabel("Absolute correction (rad)")
    axes[4].set_xlabel("Simulation time (s)")
    axes[4].legend(ncol=1, fontsize=8, loc="lower right")
    for axis in axes:
        if event_s is not None:
            axis.axvline(event_s, color="#aaaaaa", linestyle=":", alpha=.7)
        axis.grid(True, alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    note = ((f"Thread released at t={event_s:.2f} s; release alone is not successful cap removal"
             if event_s is not None else "Thread never released; commanded pose is not task success") if cap
            else "Support fully removed at t=3 s; actual pose can differ from the commanded target")
    fig.suptitle("Video reference + PPO correction: saved episode " + str(args.episode) + "\n" + note, fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    actual = trace["qpos"][:, qids]
    tracking = trace["ctrl"] - actual
    report = dict(model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  episode=args.episode, active_samples=len(time), task="cap" if cap else "cube",
                  joint_order=[model.joint(j).name for j in model.actuator_trnid[:, 0]],
                  bound_rad=args.bound, requested_max_rad=float(np.abs(trace["requested_residual"]).max()),
                  effective_max_rad=float(np.abs(trace["effective_residual"]).max()),
                  tracking_max_rad=float(np.abs(tracking).max()),
                  tracking_rmse_rad=np.sqrt(np.mean(tracking ** 2, axis=0)).tolist(),
                  last_target_rad=trace["ctrl"][-1].tolist(), last_actual_rad=actual[-1].tolist(),
                  note="One saved episode, not a success estimate. Residual bounds constrain targets, not physical tracking error.")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
