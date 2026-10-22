"""Run real BC/PPO with the S06 cylinder environment and existing audited trainer."""

import run_s05_job
from orca_s06_env import OrcaS06Grasp


if __name__ == "__main__":
    run_s05_job.OrcaS05Grasp = OrcaS06Grasp
    run_s05_job.main()

