"""Bind cylinder inputs to the unchanged S02 PPO/BC/residual training code."""

from orca_cylinder_s02_env import OrcaCylinderS02Grasp, ROOT
import run_orca_suite_job as original

INPUTS = ROOT/'artifacts/orca_cylinder_s02/inputs'


def main():
    original.MODEL = INPUTS/'model.xml'
    original.REFERENCE = INPUTS/'reference.npz'
    original.RAW_REFERENCE = INPUTS/'unrefined.npz'
    original.DATASET = INPUTS/'demos/trajectories.npz'
    original.OrcaSuiteGrasp = OrcaCylinderS02Grasp
    original.main()


if __name__ == '__main__':
    main()
