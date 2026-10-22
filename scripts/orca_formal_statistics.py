"""Seed-level estimates; evaluation episodes are not independent training runs."""

from itertools import product
import numpy as np


def seed_estimate(values, draws=20000, seed=78421):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite, nonempty per-training-seed values")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(n_training_seeds=len(values), individual=values.tolist(), mean=float(values.mean()),
                std=float(values.std(ddof=1)) if len(values) > 1 else None,
                bootstrap95=np.quantile(means, [.025, .975]).tolist(),
                interval_scope="Conditional on the fixed evaluation set; resampling training seeds only",
                caution="Five seeds give imprecise intervals; a degenerate interval is not a population certainty")


def paired_effect(left, right):
    if set(left) != set(right) or not left:
        raise ValueError("Paired comparison requires identical training seeds")
    seeds = sorted(left)
    difference = np.array([left[k] - right[k] for k in seeds], float)
    if len(seeds) > 16:
        raise ValueError("Exact sign-flip computation is limited to 16 pairs")
    null = np.mean(np.asarray(list(product((-1, 1), repeat=len(seeds)))) * difference, axis=1)
    estimate = seed_estimate(difference)
    estimate.update(seeds=seeds, direction="left minus right",
                    exact_two_sided_signflip_p=float(np.mean(np.abs(null) >= abs(difference.mean()) - 1e-12)),
                    minimum_two_sided_p=2 / 2**len(seeds))
    return estimate
