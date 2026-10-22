"""Keep frozen S05/S06 CLI behavior; add audited S07 experiment controls."""

import argparse
import functools
import sys

import run_s05_job
from orca_s07_env import OrcaS07Grasp
from orca_suite_env import BENCHMARKS


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--benchmark", choices=tuple(BENCHMARKS), default="B0_nominal")
    p.add_argument("--residual-bound", type=float, default=.08)
    p.add_argument("--residual-gain", type=float, default=1.)
    a, rest = p.parse_known_args()
    run_s05_job.OrcaS05Grasp = functools.partial(OrcaS07Grasp, **vars(a))
    sys.argv = [sys.argv[0], *rest]
    run_s05_job.main()


if __name__ == "__main__":
    main()
