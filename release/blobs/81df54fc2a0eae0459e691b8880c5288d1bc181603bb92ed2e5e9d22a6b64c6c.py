"""Full-state residual audits and independent continuous-grasp scoring."""

import numpy as np


def summarize(trace, arm, bound, low, high, max_delta=.1):
    valid = (trace["alive"] & (trace["thumb"] > 0) & (trace["contacts"] >= 2)
             & (trace["forbidden"] == 0) & (trace["released"] > 0)
             & (trace["cube_z"] > trace["spawn_z"]-.01))
    count = np.zeros(valid.shape[0], int)
    best = count.copy()
    for t in range(valid.shape[1]):
        count = np.where(valid[:, t], count+1, 0)
        best = np.maximum(best, count)
    holds = best*.05
    np.testing.assert_allclose(holds, trace["hold"].max(axis=1), atol=1e-5)
    success = (holds >= 30) & trace["alive"][:, -1] & valid[:, -1]
    assert np.isfinite(trace["ctrl"]).all()
    assert np.all(trace["ctrl"] >= np.asarray(low)-1e-5)
    assert np.all(trace["ctrl"] <= np.asarray(high)+1e-5)
    assert np.max(np.abs(np.diff(trace["ctrl"], axis=1))) <= max_delta+1e-5
    requested_max = effective_max = 0.
    if arm == "residual":
        requested = np.clip(trace["action"], -1, 1)*bound
        np.testing.assert_allclose(trace["requested_residual"], requested, atol=1e-6)
        np.testing.assert_allclose(trace["desired_ctrl"],
                                   np.clip(trace["video_base"]+requested, low, high), atol=1e-6)
        np.testing.assert_allclose(trace["effective_residual"], trace["ctrl"]-trace["video_base"], atol=1e-6)
        requested_max = float(np.max(np.abs(requested)))
        effective_max = float(np.max(np.abs(trace["effective_residual"])))
        assert requested_max <= bound+1e-5 and effective_max <= bound+1e-5
    row = dict(success30=float(success.mean()), success30_any=float(np.mean(holds >= 30)),
               mean_hold=float(holds.mean()), min_hold=float(holds.min()), max_hold=float(holds.max()),
               retained=float(trace["alive"][:, -1].mean()), mean_return=float(trace["reward"].sum(axis=1).mean()),
               requested_residual_max_rad=requested_max, effective_residual_max_rad=effective_max)
    return row, holds, success
