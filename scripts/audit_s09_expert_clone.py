"""Separate exact parameter cloning from floating-point closed-loop differences."""

import argparse
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np

from make_s09_mode_experts import verify_copies
from run_s09_mode_experts import BASE_FACTORY, network_factory
from retarget_s09_reference import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    cfg = json.loads((args.run/"run_config.json").read_text())
    conversion = json.loads((Path(cfg["checkpoint"]).parent/"conversion.json").read_text())
    source = Path(conversion["source_checkpoint"])
    folder = args.run/"step_000000000"
    before, after = [pickle.loads(p.read_bytes()) for p in (source, folder/"params.pkl")]
    verify_copies(before, after)
    assert digest(source) == conversion["source_checkpoint_sha256"]
    with np.load(folder/"preview_trajectories.npz") as z:
        obs = z["obs"][z["active"].astype(bool)]
    inputs = dict(state=jp.asarray(obs), privileged_state=jp.pad(jp.asarray(obs), ((0, 0), (0, 17))))
    values = []
    for factory, params in ((BASE_FACTORY, before), (network_factory, after)):
        net = factory(dict(state=(123,), privileged_state=(140,)), 17)
        fn = jax.jit(lambda x:net.parametric_action_distribution.mode(
            net.policy_network.apply(params[0], params[1], x)))
        values.append(np.asarray(fn(inputs)))
    error = float(np.max(np.abs(values[0]-values[1])))
    assert np.isfinite(values).all() and error <= 1e-5
    report = dict(status="verified", backend=jax.default_backend(), parameters_exactly_cloned=True,
        same_input_states=len(obs), max_normalized_action_error=error, tolerance=1e-5,
        source_checkpoint_sha256=digest(source), clone_checkpoint_sha256=digest(folder/"params.pkl"),
        trace_sha256=digest(folder/"preview_trajectories.npz"), source_sha256=digest(Path(__file__)),
        scope="Same-input actor comparison on saved preview states; GPU graph rounding can still diverge in contact-rich closed-loop trajectories. Not proof of bitwise identical physics.")
    (folder/"clone_behavior_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
