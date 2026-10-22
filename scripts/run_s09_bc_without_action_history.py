"""Controlled feature ablation: keep physical feedback, exclude control history.

The initialized input rows are zero and stay zero during BC. Deployment uses
the same 123-dimensional interface and a full-target network, not a residual.
"""

import json
from pathlib import Path
import sys

import jax.numpy as jp
import numpy as np

import run_s09_recovery_bc as recovery


INPUT_MASK = np.ones(123, dtype=bool)
INPUT_MASK[34:68] = False


def mask_arrays(arrays, phase_only):
    if phase_only:
        raise ValueError("Do not combine the independent feature ablations")
    return tuple(x*jp.asarray(INPUT_MASK) if i%2 == 0 else x for i,x in enumerate(arrays))


if __name__ == "__main__":
    original = recovery.mask_arrays
    recovery.mask_arrays = mask_arrays
    try:
        recovery.main()
    finally:
        recovery.mask_arrays = original
    output = Path(sys.argv[sys.argv.index("--output")+1])
    file = output/"recovery_bc_config.json"
    cfg = json.loads(file.read_text())
    cfg.update(input_columns=np.flatnonzero(INPUT_MASK).tolist(),
        excluded_columns=list(range(34,68)), intervention="No actuator-control or previous-action input during BC")
    with (output/"step_000000000/params.pkl").open("rb") as stream:
        params = recovery.pickle.load(stream)
    np.testing.assert_array_equal(np.asarray(params[1]["params"]["hidden_0"]["kernel"])[~INPUT_MASK], 0)
    cfg["excluded_input_weights_verified_zero"] = True
    file.write_text(json.dumps(cfg, indent=2)+"\n")
