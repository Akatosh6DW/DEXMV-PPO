"""Derive diagnostic expert copies with scaled non-phase input connections."""

import argparse
import copy
import json
from pathlib import Path
import pickle

import jax
import numpy as np

from retarget_s09_reference import digest
from run_s09_bc_phase_curriculum import KEEP
from run_s09_mode_experts import ARCHITECTURE


def scaled_experts(params, gain):
    if not np.isfinite(gain) or not 0 <= gain <= 1:
        raise ValueError("Feedback gain must be finite and within [0, 1]")
    result = copy.deepcopy(jax.device_get(params))
    if len(result[1]["experts"]) != 6:
        raise ValueError("Six experts required")
    for expert in result[1]["experts"]:
        kernel = np.asarray(expert["params"]["hidden_0"]["kernel"]).copy()
        if kernel.shape[0] != len(KEEP):
            raise ValueError("Unexpected actor input")
        kernel[~KEEP] *= gain
        expert["params"]["hidden_0"]["kernel"] = kernel
    left, spec = jax.tree_util.tree_flatten_with_path(params)
    right, other = jax.tree_util.tree_flatten_with_path(result)
    assert spec == other
    changed = 0
    for (path, old), (_, new) in zip(left, right):
        keys = tuple(getattr(k, "key", getattr(k, "idx", None)) for k in path)
        old, new = np.asarray(old), np.asarray(new)
        if len(keys) == 6 and keys[:2] == (1, "experts") and keys[3:] == ("params", "hidden_0", "kernel"):
            np.testing.assert_array_equal(old[KEEP], new[KEEP])
            np.testing.assert_array_equal(old[~KEEP]*gain, new[~KEEP])
            changed += int(np.count_nonzero(old != new))
        else:
            np.testing.assert_array_equal(old, new)
    return result, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gain", type=float, required=True)
    args = parser.parse_args()
    parent = args.source.parent if (args.source.parent/"run_config.json").exists() else args.source.parent.parent
    config = json.loads((parent/"run_config.json").read_text())
    assert config["policy_architecture"] == ARCHITECTURE and config["method"] == "guided"
    before_hash = digest(args.source)
    result, changed = scaled_experts(pickle.loads(args.source.read_bytes()), args.gain)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output/"params.pkl"
    checkpoint.write_bytes(pickle.dumps(result))
    compatibility = {k:config[k] for k in ("model_sha256", "reference_sha256", "method", "physics_backend",
        "actor_dim", "critic_dim", "env_config", "policy_architecture")}
    compatibility.update(diagnostic_only=True, source_checkpoint_sha256=before_hash)
    (args.output/"run_config.json").write_text(json.dumps(compatibility, indent=2)+"\n")
    record = dict(status="derived_not_trained", gain=args.gain, source_checkpoint=str(args.source.resolve()),
        source_checkpoint_sha256=before_hash, derived_checkpoint_sha256=digest(checkpoint), changed_parameters=changed,
        other_parameters_unchanged=True, policy_updates=0, source_sha256=digest(Path(__file__)),
        caveat="Post-hoc input-gain intervention, not a learned stability constraint or deployable validation")
    (args.output/"feedback_gain.json").write_text(json.dumps(record, indent=2)+"\n")
    assert digest(args.source) == before_hash
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
