# 中文阅读副本：07 残差合成公式
# 原始入口：scripts/orca_residual_controls.py
# 仅用于阅读，保留原路径假设；不要从此副本启动训练。运行请使用原始入口。
# 本副本只增加注释，生成时核验AST与非注释词法序列完全一致。
"""Reference preprocessing and bounded, non-integrating residual composition."""

from pathlib import Path

import numpy as np


# 导读：检查时间及关节范围，将离线参考插值到控制时刻；末帧后保持末值。
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
    # 导读：预处理参考本身的变化速率；实际执行的合成目标还要经过环境的统一速率限制。
    for target in raw:
        previous = np.clip(target, previous-max_delta, previous+max_delta)
        base.append(previous.copy())
    return np.asarray(base, np.float32), raw.astype(np.float32)


# 导读：真正的残差架构在此：参考角+有界修正。每步独立相加，不累计成无限漂移。
def compose(base, action, bound, low, high, xp=np):
    residual = xp.clip(action, -1, 1)*bound
    desired = xp.clip(base+residual, low, high)
    return desired, residual
