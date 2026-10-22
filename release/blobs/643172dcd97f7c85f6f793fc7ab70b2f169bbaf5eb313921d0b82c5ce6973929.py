"""Condition full-target BC+PPO on an initial-pose grasp-mode label.

Only six template AXES are used at inference, never template control targets.
The network still outputs all 17 absolute targets; this is not residual control.
"""

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np

import run_s09_bc_repair as repair
from orca_s09_env import OrcaS09Grasp, cylinder_axis
from retarget_s09_reference import digest


class ModeConditionedGrasp(OrcaS09Grasp):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        if self.arm != "pure":
            raise ValueError("Full-target policy only")
        with np.load(Path(model).parent/"reference.npz", allow_pickle=False) as z:
            quats = z["object_quaternion"]
        if len(quats) != 6:
            raise ValueError("This version uses six pose modes")
        self.reference_axes = jax.vmap(cylinder_axis)(jp.asarray(quats, jp.float32))
        with self._config.unlocked():
            self._config.observation_variant = "initial_pose_mode_onehot_95_101"
            self._config.action_variant = "full_absolute_target_no_reference_action"

    def _obs(self, data, info):
        obs = super()._obs(data, info)
        actor = obs["state"].at[95:101].set(jax.nn.one_hot(info["reference_index"], 6))
        return dict(state=actor, privileged_state=jp.concatenate([actor, data.actuator_force]))


def condition_arrays(path, axes, acquisition_steps=30):
    manifest = json.loads((Path(path).parent/"dataset_manifest.json").read_text())
    with np.load(path, allow_pickle=False) as z:
        obs, action, active = z["obs"].copy(), z["action"], z["active"].astype(bool)
    # The mode depends on the measured initial pose, not a success label or test result.
    indices = np.argmax(np.abs(obs[:, 0, 74:77] @ axes.T), axis=1)
    obs[..., 95:101] = np.eye(6, dtype=np.float32)[indices, None, :]
    split = manifest.get("train_episodes", len(obs)*3//4)
    training = np.arange(len(obs))[:, None] < split
    early = np.arange(obs.shape[1])[None, :] < acquisition_steps
    masks = (active & training & early, active & training & ~early, active & ~training)
    if not all(mask.any() for mask in masks):
        raise ValueError("Missing training/validation samples")
    return tuple(jp.asarray(x) for mask in masks for x in (obs[mask], action[mask]))


def main():
    if "--repair-observation" in sys.argv or "--repair-acquisition-steps" in sys.argv:
        raise ValueError("Do not silently combine independent observation/sampling interventions")
    inp = Path(sys.argv[sys.argv.index("--input")+1])
    output = Path(sys.argv[sys.argv.index("--output")+1])
    if "--checkpoint" in sys.argv:
        parent = Path(sys.argv[sys.argv.index("--checkpoint")+1]).parent
        if not (parent/"run_config.json").exists():
            parent = parent.parent
        old = json.loads((parent/"run_config.json").read_text())
        if old["env_config"].get("observation_variant") != "initial_pose_mode_onehot_95_101":
            raise ValueError("Conditioned checkpoint requires the matching measured-pose observation interface")
    with np.load(inp/"reference.npz", allow_pickle=False) as z:
        axes = np.asarray(jax.vmap(cylinder_axis)(jp.asarray(z["object_quaternion"], jp.float32)))
    original = repair.runner.dataset_arrays

    def validated_arrays(*args, **kwargs):
        original(*args, **kwargs)
        return condition_arrays(args[0], axes)

    repair.runner.dataset_arrays = validated_arrays
    repair.runner.OrcaS09Grasp = ModeConditionedGrasp
    try:
        repair.main()
    finally:
        if output.exists():
            snapshot = output/"sources"/Path(__file__).name
            (output/"conditioning.json").write_text(json.dumps(dict(
                observation_mode="six initial-pose classes, measured at reset and remembered",
                axes=axes.tolist(), action_architecture="learned full target, no reference addition",
                limitations="Additional pose conditioning versus original BC baseline; not a like-for-like observation ablation",
                source_sha256=digest(snapshot if snapshot.exists() else Path(__file__))), indent=2)+"\n")


if __name__ == "__main__":
    main()
