"""Balance present pose modes within each existing BC temporal stratum.

Only training rows are repeated. No missing-mode demonstration is invented and
validation rows are unchanged. The existing fitter, Adam settings, phase mask,
network, and absolute-target interface are reused without modification.
"""

import json
import math
from pathlib import Path
import sys
import time

import jax.numpy as jp
import numpy as np

import run_s09_bc_phase_curriculum as phase
from retarget_s09_reference import digest


def balance_stratum(obs, actions, max_rows=1000000):
    obs, actions = np.asarray(obs), np.asarray(actions)
    if obs.ndim != 2 or obs.shape[1] != 123 or actions.shape != (len(obs), 17):
        raise ValueError("Expected paired 123-input, 17-output BC rows")
    modes = obs[:, 95:101]
    if not np.isin(modes, (0, 1)).all() or not np.all(modes.sum(axis=1) == 1):
        raise ValueError("Expected measured-pose one-hot conditions")
    labels = modes.argmax(axis=1)
    counts = np.bincount(labels, minlength=6)
    present = np.flatnonzero(counts)
    if len(present) < 2:
        raise ValueError("Mode balancing requires at least two represented modes")
    common = math.lcm(*(int(counts[i]) for i in present))
    if common * len(present) > max_rows:
        raise ValueError("Unequal row counts cause excessive exact replication")
    indices = np.concatenate([np.tile(np.flatnonzero(labels == mode), common//int(counts[mode]))
                              for mode in present])
    metadata = dict(original_rows=len(obs), balanced_rows=len(indices),
        original_mode_counts=counts.tolist(), present_modes=present.tolist(),
        rows_per_present_mode=common, absent_modes=np.flatnonzero(counts == 0).tolist())
    return jp.asarray(obs[indices]), jp.asarray(actions[indices]), metadata


def balance_arrays(arrays):
    if len(arrays) != 6:
        raise ValueError("Use acquisition, holding and untouched validation pairs")
    early_obs, early_actions, early = balance_stratum(*arrays[:2])
    late_obs, late_actions, late = balance_stratum(*arrays[2:4])
    if early["present_modes"] != late["present_modes"]:
        raise ValueError("Temporal strata do not represent the same modes")
    return (early_obs, early_actions, late_obs, late_actions, *arrays[4:]), dict(early=early, late=late)


def main():
    if "--checkpoint" in sys.argv or "--dataset" not in sys.argv:
        raise ValueError("This probe retrains BC from the original fixed dataset")
    if "--steps" not in sys.argv or int(sys.argv[sys.argv.index("--steps")+1]) != 0:
        raise ValueError("Evaluate the BC origin first; run PPO separately afterwards")
    output = Path(sys.argv[sys.argv.index("--output")+1])
    runner = phase.conditioned.repair.runner
    original_fit = runner.fit_matched_bc

    def fit(net, params, arrays, updates, seed):
        started = time.monotonic()
        balanced, metadata = balance_arrays(arrays)
        fitted, rows = original_fit(net, params, balanced, updates, seed)
        metadata.update(bc_updates=updates, seed=seed, fit_wall_sec=time.monotonic()-started,
            sampler="uniform over exactly replicated present modes within each temporal stratum",
            early_late_batch_sizes=[128, 128], validation_rows_unchanged=True,
            new_demonstrations_added=False, missing_mode_invented=False,
            original_fitter_sha256=digest(Path(sys.modules[original_fit.__module__].__file__)),
            source_sha256=digest(Path(__file__)))
        (output/"bc_mode_balance.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return fitted, rows

    runner.fit_matched_bc = fit
    try:
        phase.main()
    finally:
        runner.fit_matched_bc = original_fit


if __name__ == "__main__":
    main()
