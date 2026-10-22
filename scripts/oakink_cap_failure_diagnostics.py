"""Describe audited cap failures without modifying rewards, scores, or episodes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def classify(trace):
    records = []
    for i in range(len(trace["alive"])):
        active = trace["active"][i].astype(bool)
        alive = trace["alive"][i].astype(bool)
        last = int(np.flatnonzero(active)[-1])
        released = np.flatnonzero((trace["released"][i] > 0) & alive)
        stops = np.flatnonzero(active & ~alive)
        forbidden = np.flatnonzero((trace["forbidden"][i] > 0) & active)
        penetration = np.flatnonzero((trace["hand_penetration"][i] > .005) & active)
        initial_invalid = bool(trace["reset_penetration"][i, 0] > .003)
        final_paired = bool(trace["paired"][i, last])
        final_clearance = bool(trace["lift"][i, last] >= .018)
        bottle = bool(trace["bottle_contact"][i, last])
        floor = bool(trace["floor_contact"][i, last])
        success = bool(trace["success"][i, -1] and alive[-1])
        if initial_invalid:
            reason = "invalid_initial_geometry"
        elif success:
            reason = "success"
        elif len(forbidden):
            reason = "disqualified_by_forbidden_contact"
        elif len(penetration):
            reason = "disqualified_by_hand_penetration"
        elif not len(released):
            reason = "never_released_during_live_episode"
        elif len(stops):
            reason = "released_then_floor_termination" if floor else "released_then_other_termination"
        elif bottle or floor:
            reason = "released_but_supported"
        elif not final_paired:
            reason = "released_but_no_final_paired_grasp"
        elif not final_clearance:
            reason = "released_but_insufficient_final_clearance"
        else:
            reason = "released_but_insufficient_final_hold_duration"
        records.append(dict(reason=reason, success=success,
            released_during_live_episode=bool(len(released)),
            release_time_s=float((released[0] + 1) * .05) if len(released) else None,
            termination_time_s=float((stops[0] + 1) * .05) if len(stops) else None,
            forbidden_ever=bool(len(forbidden)), excessive_penetration_ever=bool(len(penetration)),
            invalid_initial=initial_invalid, alive_at_end=bool(alive[-1]),
            last_active_time_s=float((last + 1) * .05), last_active_paired_grasp=final_paired,
            last_active_clearance_m=float(trace["lift"][i, last]),
            last_active_bottle_contact=bottle, last_active_floor_contact=floor,
            peak_force_n=float(trace["force"][i, active].max()),
            longest_valid_hold_s=float(np.where(alive, trace["hold_sec"][i], 0).max())))
    return dict(counts=dict(Counter(r["reason"] for r in records)), episodes=records,
        note="Descriptive hierarchy only; flags preserve overlapping failure causes. Uses only active/alive states, not post-termination physics. No score or success-threshold changes.")


def from_file(path):
    keys = ("alive", "active", "released", "forbidden", "hand_penetration", "reset_penetration",
            "paired", "lift", "bottle_contact", "floor_contact", "success", "force", "hold_sec")
    with np.load(path) as packet:
        result = classify({key: packet[key] for key in keys})
    result["trace_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = from_file(args.trace)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()
