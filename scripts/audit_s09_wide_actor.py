"""Bind widened full-target inference to the unchanged native replay audit."""

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
from brax.training.networks import FeedForwardNetwork

import audit_s09_bc_repair
from run_s09_wide_actor import ARCHITECTURE, wide_networks, runner


def matched_actor(actor, batch_size):
    """Keep the evaluator's per-frame GEMM shape instead of flattening time."""
    if batch_size <= 0:
        raise ValueError("Positive evaluation batch size required")

    def apply(normalizer, params, observations):
        values = observations["state"]
        if values.ndim != 3:
            return actor.apply(normalizer, params, observations)
        cases, _, _ = values.shape
        if cases > batch_size:
            raise ValueError("Saved preview batch exceeds evaluation batch")
        frames = jp.pad(values.transpose(1, 0, 2), ((0,0),(0,batch_size-cases),(0,0)))
        def frame(_, obs):
            return None, jax.vmap(lambda x: actor.apply(normalizer, params, dict(state=x)))(obs)
        _, logits = jax.lax.scan(frame, None, frames)
        return logits[:, :cases].transpose(1, 0, 2)

    return FeedForwardNetwork(init=actor.init, apply=apply)


if __name__ == "__main__":
    run = Path(sys.argv[sys.argv.index("--run")+1])
    cfg = json.loads((run/"run_config.json").read_text())
    if cfg.get("policy_architecture") != ARCHITECTURE:
        raise ValueError("Require the widened full-target policy")
    with wide_networks():
        original = runner.initialize_matched
        def initialize(*args, **kwargs):
            net, params = original(*args, **kwargs)
            return net.replace(policy_network=matched_actor(net.policy_network, cfg["eval_batch"])), params
        runner.initialize_matched = initialize
        try:
            audit_s09_bc_repair.main()
        finally:
            runner.initialize_matched = original
    step = int(sys.argv[sys.argv.index("--step")+1]) if "--step" in sys.argv else 0
    path = run/f"step_{step:09d}"/"native_replay_audit.json"
    record = json.loads(path.read_text())
    record["inference_shape"] = f"per-frame scan(vmap(actor)), padded to evaluation batch {cfg['eval_batch']}; unchanged 1e-5 tolerance"
    path.write_text(json.dumps(record, indent=2)+"\n")
