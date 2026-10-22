"""BC data validation for raw replay, including genuinely unsuccessful teachers.

Unlike the old expert-data loader, this does not require a successful holding
phase. Missing late samples are not synthesized and terminated samples are not
used. Both sampler slots contain the same training pool, so the existing equal
weighted BC loss is simply uniform imitation over actual active samples.
"""

import hashlib
import json
from pathlib import Path

import jax.numpy as jp
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_dataset_arrays(path, model_hash, reference_hash, physics_backend):
    manifest = json.loads((path.parent/"dataset_manifest.json").read_text())
    expected = dict(dataset_sha256=digest(path), model_sha256=model_hash,
                    reference_sha256=reference_hash, actor_dim=123, physics_backend=physics_backend)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Dataset is not bound to this fresh task")
    with np.load(path, allow_pickle=False) as data:
        obs, action, active = data["obs"], data["action"], data["active"].astype(bool)
    if (obs.ndim != 3 or obs.shape[-1] != 123 or len(obs) < 4
            or active.shape != obs.shape[:2] or action.shape != (*obs.shape[:2], 17)):
        raise ValueError("Invalid replay array shapes")
    if (not np.isfinite(obs).all() or not np.isfinite(action).all()
            or np.any(obs[..., 94] != 0) or np.any(obs[..., 95:112] != 0)
            or np.any(np.abs(action) > 1.000001)):
        raise ValueError("Invalid values or reference/privileged observation leak")
    split = len(obs)*3//4
    mask = np.arange(len(obs))[:, None] < split
    train, validation = active & mask, active & ~mask
    if not train.any() or not validation.any():
        raise ValueError("No active training/validation samples")
    arrays = tuple(jp.asarray(x) for x in (obs[train], action[train], obs[validation], action[validation]))
    train_obs, train_action, val_obs, val_action = arrays
    return train_obs, train_action, train_obs, train_action, val_obs, val_action
