"""Execute frozen conditional BC stochastically, with NO parameter updates."""

import argparse
import json
from pathlib import Path
import sys

import run_s09_bc_conditioned as conditioned
from run_s09_bc_trust import with_std


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--probe-std", type=float, required=True)
    a, rest = p.parse_known_args()
    if "--steps" not in rest or rest[rest.index("--steps")+1] != "0":
        p.error("Only explicit --steps 0 is supported")
    output = Path(rest[rest.index("--output")+1])
    original = conditioned.repair.runner.networks.make_inference_fn
    def factory(net):
        make = original(net)
        def build(params, deterministic=True):
            return make(with_std(params,a.probe_std), deterministic=False)
        return build
    conditioned.repair.runner.networks.make_inference_fn = factory
    sys.argv = [sys.argv[0], *rest]
    try:
        conditioned.main()
    finally:
        if output.exists():
            (output/"stochastic_probe.json").write_text(json.dumps(dict(
                optimizer_updates=0, deterministic=False, sampling_std=a.probe_std,
                mean_unchanged=True, saved_params_are_original_checkpoint=True,
                reproduction="Use this script plus probe-std; do not treat saved params as a deterministic playback"),indent=2)+"\n")


if __name__ == "__main__":
    main()
