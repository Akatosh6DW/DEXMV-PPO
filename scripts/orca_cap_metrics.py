"""Offline whole-episode cap criteria, including violations after a hold."""

import numpy as np


def hold_seconds(trace, dt=.05, target_angle=np.pi/2, thread_tolerance=.0002):
    bad = (trace["forbidden"] > 0) | (trace["thread_error"] > .001) | ~trace["alive"]
    invalid_prefix = np.maximum.accumulate(bad, axis=1)
    valid = ((trace["angle"] >= target_angle) & (trace["paired"] > 0)
             & (trace["thread_error"] < thread_tolerance) & ~invalid_prefix)
    count = np.zeros(valid.shape[0], dtype=int)
    best = count.copy()
    for t in range(valid.shape[1]):
        count = np.where(valid[:, t], count+1, 0)
        best = np.maximum(best, count)
    raw_holds = best*dt
    if "best_hold_sec" in trace:
        np.testing.assert_allclose(raw_holds, np.max(np.where(trace["alive"], trace["best_hold_sec"], 0), axis=1), atol=1e-5)
    # A later violation disqualifies the whole episode, including earlier holds.
    return np.where(np.any(bad, axis=1), 0., raw_holds)


def summarize(trace, env):
    holds = hold_seconds(trace, env.dt, env._config.target_angle, env._config.thread_tolerance)
    row = dict(success=float(np.mean(holds >= env._config.hold_success_sec)),
               mean_hold=float(holds.mean()), max_hold=float(holds.max()),
               mean_final_degrees=float(trace["degrees"][:, -1].mean()),
               mean_max_degrees=float(trace["degrees"].max(axis=1).mean()),
               max_thread_error_m=float(trace["thread_error"].max()),
               forbidden_episode_fraction=float(np.mean(np.any(trace["forbidden"] > 0, axis=1))),
               mean_paired_fraction=float(trace["paired"].mean()),
               peak_force_n=float(trace["force"].max()),
               mean_return=float(trace["reward"].sum(axis=1).mean()))
    return row, holds
