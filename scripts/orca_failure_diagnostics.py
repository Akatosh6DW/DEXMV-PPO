"""Describe why a held-out cube episode failed without changing its score."""

from collections import Counter

import numpy as np


def classify(trace):
    records = []
    for i in range(len(trace["alive"])):
        alive = trace["alive"][i].astype(bool)
        hold = trace["hold"][i]
        acquired = np.flatnonzero((hold > 0) & alive)
        stopped = np.flatnonzero(~alive)
        final_height = bool(trace["cube_z"][i, -1] > trace["spawn_z"][i, -1] - .01)
        final_contacts = bool(trace["thumb"][i, -1] > 0 and trace["contacts"][i, -1] >= 2 and trace["forbidden"][i, -1] == 0)
        if trace["initial_penetration_m"][i, 0] > .003:
            reason = "invalid_initial_geometry"
        elif alive[-1] and hold[-1] > 0 and hold.max() >= 30:
            reason = "success"
        elif len(stopped) and not trace["released"][i, stopped[0]]:
            reason = "terminated_before_support_removed"
        elif not len(acquired):
            reason = "never_acquired_valid_unsupported_hold"
        elif not alive[-1]:
            reason = "terminated_after_acquisition"
        elif not final_height:
            reason = "retained_but_height_invalid"
        elif not final_contacts:
            reason = "retained_but_contact_invalid"
        else:
            reason = "insufficient_uninterrupted_duration"
        losses = np.flatnonzero((hold[:-1] > 0) & (hold[1:] == 0)) + 1
        records.append(dict(reason=reason, acquisition_time_s=float((acquired[0] + 1) * .05) if len(acquired) else None,
                            first_hold_loss_time_s=float((losses[0] + 1) * .05) if len(losses) else None,
                            termination_time_s=float((stopped[0] + 1) * .05) if len(stopped) else None,
                            alive_at_end=bool(alive[-1]), final_height_valid=final_height, final_contacts_valid=final_contacts))
    return dict(counts=dict(Counter(r["reason"] for r in records)), episodes=records,
                note="Termination is not automatically labelled dropping; a retained object may fail height/contact/duration criteria")


def from_file(path):
    keys = ("alive", "hold", "released", "thumb", "contacts", "forbidden", "cube_z", "spawn_z", "initial_penetration_m")
    with np.load(path) as packet:
        return classify({key: packet[key] for key in keys})
