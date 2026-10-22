"""BC first learns the teacher's mode/phase law; PPO can learn full feedback.

BC input rows unrelated to mode/phase are zeroed, including their first-layer
weights. They are NOT frozen during PPO. At deployment all observations enter
one trainable network; no reference targets are replayed or added.
"""

import copy
import json
from pathlib import Path
import sys

import jax.numpy as jp
import numpy as np

import run_s09_bc_conditioned as conditioned
from retarget_s09_reference import digest


KEEP = np.zeros(123, dtype=bool)
KEEP[93:113] = True


def phase_initialization(params):
    params = copy.deepcopy(params)
    first = params[1]["params"]["hidden_0"]
    if first["kernel"].shape[0] != 123:
        raise ValueError("Unexpected policy input interface")
    first["kernel"] = jp.asarray(first["kernel"]) * jp.asarray(KEEP)[:,None]
    return params


def main():
    runner = conditioned.repair.runner
    original_fit = runner.fit_matched_bc
    output = Path(sys.argv[sys.argv.index("--output")+1])
    def fit(net, params, arrays, updates, seed):
        masked = tuple(x * jp.asarray(KEEP) if i % 2 == 0 else x for i,x in enumerate(arrays))
        fitted, rows = original_fit(net, phase_initialization(params), masked, updates, seed)
        np.testing.assert_array_equal(np.asarray(fitted[1]["params"]["hidden_0"]["kernel"])[~KEEP], 0)
        return fitted, rows
    runner.fit_matched_bc = fit
    try:
        conditioned.main()
    finally:
        if output.exists():
            snapshot = output/"sources"/Path(__file__).name
            (output/"phase_curriculum.json").write_text(json.dumps(dict(
                bc_input_columns=np.flatnonzero(KEEP).tolist(),
                ppo_input_columns="all123, all policy weights trainable",
                reference_action_added=False, teacher_law="initial measured pose mode + elapsed phase",
                limitation="BC pretraining itself is feedforward mode/phase imitation, not state-feedback recovery",
                source_sha256=digest(snapshot if snapshot.exists() else Path(__file__))),indent=2)+"\n")


if __name__ == "__main__":
    main()
