"""Development-only stochastic execution of frozen BC; no optimizer updates."""

import argparse
import copy
import json
from pathlib import Path
import sys

import jax.numpy as jp
import numpy as np

import run_s07_job
import run_s05_job


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-std", type=float, required=True)
    parser.add_argument("--probe-task", choices=("cube", "cylinder"), default="cylinder")
    args, rest = parser.parse_known_args()
    if args.probe_std <= .001:
        parser.error("NormalTanh minimum std is .001")
    if "--steps" not in rest or rest[rest.index("--steps")+1] != "0":
        parser.error("This is evaluation only; explicit --steps 0 required")
    output = Path(rest[rest.index("--output")+1])
    original = run_s05_job.networks.make_inference_fn

    def stochastic_factory(net):
        factory = original(net)
        def build(params, deterministic=True):
            changed = copy.deepcopy(params)
            head = changed[1]["params"]["hidden_2"]
            head["kernel"] = jp.asarray(head["kernel"]).at[:, 17:].set(0)
            head["bias"] = jp.asarray(head["bias"]).at[17:].set(np.log(np.expm1(args.probe_std-.001)))
            return factory(changed, deterministic=False)
        return build

    run_s05_job.networks.make_inference_fn = stochastic_factory
    sys.argv = [sys.argv[0], *rest]
    (run_s05_job.main if args.probe_task == "cube" else run_s07_job.main)()
    metadata = dict(probe_std=args.probe_std, task=args.probe_task, deterministic=False, optimizer_updates=0,
        input_policy_mean_unchanged=True, saved_params_are_input_checkpoint=True,
        note="Only sampling standard deviation overridden at inference. Saved params retain original checkpoint; reproduce with this script and --probe-std. Development seeds, not independent benchmark.")
    (output / "noise_probe.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
