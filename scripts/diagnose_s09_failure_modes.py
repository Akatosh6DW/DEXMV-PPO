"""Decompose frozen validity gates on saved previews without changing labels."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from retarget_s09_reference import digest


def episode_reasons(trace, initial_qpos, object_qpos_adr):
    audit = np.asarray(trace["audit"], dtype=np.float64)
    active = np.asarray(trace["active"], dtype=bool)
    released = active & (audit[:, 10] > 0)
    valid = released & (audit[:, 11] > 0)
    breaks = np.r_[False, valid[:-1]] & released & ~valid
    height_loss = float(initial_qpos[object_qpos_adr+2]) - np.asarray(
        trace["qpos"][:, object_qpos_adr+2], dtype=np.float64)
    # Counts overlap intentionally: one frame can violate several gates.
    reasons = dict(
        thumb_load_le_0_02n=audit[:, 0] <= .02,
        opposing_load_le_0_02n=np.max(audit[:, 1:5], axis=1) <= .02,
        height_loss_ge_15mm=height_loss >= .015,
        penetration_gt_3mm=audit[:, 6] > .003,
        forbidden_contact=audit[:, 5] > 0,
        initial_penetration_gt_1mm=np.asarray(trace["initial_penetration"]) > .001,
        rope_force_nonzero=audit[:, 8] != 0,
        no_longer_alive=audit[:, 9] <= 0)
    reconstructed = released & ~np.logical_or.reduce(list(reasons.values()))
    mismatch = released & (reconstructed != valid)
    invalid = released & ~valid
    result = dict(success30=bool(trace["success30"][-1]),
        active_duration_sec=float(active.sum() * .05),
        best_hold_sec=float(np.max(trace["best_hold"])),
        released_invalid_sec=float(invalid.sum() * .05),
        valid_streak_breaks=int(breaks.sum()),
        valid_reconstruction_disagreement_frames=int(mismatch.sum()),
        unreconstructed_breaks=int(np.sum(breaks & ~np.logical_or.reduce(list(reasons.values())))),
        invalid_seconds_by_gate={name:float(np.sum(invalid & flag) * .05) for name,flag in reasons.items()},
        break_events_by_gate={name:int(np.sum(breaks & flag)) for name,flag in reasons.items()},
        max_released_height_loss_m=float(np.max(height_loss[released], initial=0)),
        max_released_depth_m=float(np.max(audit[released, 6], initial=0)),
        ever_released_forbidden=bool(np.any(released & reasons["forbidden_contact"])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / "run_config.json").read_text())
    folder = args.run / f"step_{args.step:09d}"
    model_path = Path(config["input"]) / "model.xml"
    if digest(model_path) != config["model_sha256"]:
        raise ValueError("Task model changed")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    oid = int(model.jnt_qposadr[model.joint("cube_freejoint").id])
    with np.load(folder / "initial_states.npz", allow_pickle=False) as archive:
        initial = {int(seed):qpos for seed,qpos in zip(archive["seeds"], archive["qpos"])}
    with np.load(folder / "preview_trajectories.npz", allow_pickle=False) as archive:
        trace = dict(archive)
    rows = []
    for i, seed in enumerate(trace["seeds"]):
        row = episode_reasons({k:v[i] for k,v in trace.items() if k != "seeds"}, initial[int(seed)], oid)
        rows.append(dict(seed=int(seed), **row))
    result = dict(run=str(args.run.resolve()), step=args.step,
        scope="Saved previews only; gate counts overlap; original success labels unchanged",
        precision_note="Float32 saved states can disagree at exact gate boundaries; disagreements are reported, not hidden",
        trace_sha256=digest(folder / "preview_trajectories.npz"),
        initial_states_sha256=digest(folder / "initial_states.npz"), rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
