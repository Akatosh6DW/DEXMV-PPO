"""Run disjoint predeclared S07 jobs; incomplete directories never overwritten."""

import argparse

from run_s07_study import OUT, S06, prepare, launch, record_training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("ppo", "guided", "residual"), required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    a = p.parse_args()
    if not set(a.seeds) <= {7101, 7102, 7103}:
        p.error("Only predeclared training seeds are allowed")
    prepare()
    for seed in a.seeds:
        directory = OUT / "train" / f"{a.method}_s{seed}"
        args = ["--input", OUT / "inputs", "--method", a.method, "--seed", seed,
                "--steps", 163840, "--evals", 9, "--eval-count", 8, "--eval-start", 70000,
                "--learning-rate", 1e-4, "--std", .1]
        if a.method == "guided":
            args += ["--dataset", S06 / "D03_implicit_demonstrations/trajectories.npz", "--bc-updates", 20000]
        launch(directory, args)
        if not (directory / "iteration_audit.json").exists():
            record_training(directory)


if __name__ == "__main__":
    main()
