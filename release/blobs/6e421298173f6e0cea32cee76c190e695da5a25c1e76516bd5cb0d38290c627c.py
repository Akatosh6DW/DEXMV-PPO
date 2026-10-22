"""Reference preprocessing and bounded, non-integrating residual composition."""

from pathlib import Path

import numpy as np


def load_reference(path, low, high, initial, dt, steps, max_delta):
    with np.load(Path(path), allow_pickle=False) as z:
        ctrl = np.asarray(z["ctrl"], dtype=float)
        times = np.asarray(z["time"], dtype=float)
    if ctrl.ndim != 2 or ctrl.shape[1] != len(low) or len(ctrl) < 2:
        raise ValueError("Reference must have shape (T>=2, number of actuators)")
    if times.shape != (len(ctrl),) or not np.isfinite(ctrl).all() or not np.isfinite(times).all():
        raise ValueError("Invalid reference values/timestamps")
    if abs(times[0]) > 1e-8 or np.any(np.diff(times) <= 0):
        raise ValueError("Reference timestamps must start at zero and strictly increase")
    if np.any(ctrl < np.asarray(low)-1e-5) or np.any(ctrl > np.asarray(high)+1e-5):
        raise ValueError("Reference exceeds this hand's actuator limits")
    ctrl = np.clip(ctrl, low, high)
    sample_times = (np.arange(steps)+1)*dt
    raw = np.stack([np.interp(sample_times, times, ctrl[:, j]) for j in range(ctrl.shape[1])], axis=1)
    previous = np.asarray(initial, float).copy()
    base = []
    for target in raw:
        previous = np.clip(target, previous-max_delta, previous+max_delta)
        base.append(previous.copy())
    return np.asarray(base, np.float32), raw.astype(np.float32)


def compose(base, action, bound, low, high, xp=np):
    residual = xp.clip(action, -1, 1)*bound
    desired = xp.clip(base+residual, low, high)
    return desired, residual
