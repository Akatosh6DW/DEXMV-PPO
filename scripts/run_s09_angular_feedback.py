"""Add angular velocity to previously zero full-target observation slots.

This is a PPO-input intervention, not new BC supervision or residual control.
Only full-target mode-conditioned policies can use slots 101:104; residual
policies use those slots for their base targets and are explicitly rejected.
"""

import argparse
import copy
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jp
import numpy as np

import run_s09_constrained_ppo as constrained
from run_s09_bc_conditioned import ModeConditionedGrasp
from retarget_s09_reference import digest

FEATURE = "mujoco_freejoint_angular_qvel_x0.1_clip5_slots101_104"


class AngularFeedbackGrasp(ModeConditionedGrasp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self._config.unlocked():
            self._config.angular_velocity_observation = FEATURE

    def _obs(self, data, info):
        original = super()._obs(data, info)
        actor = original["state"].at[101:104].set(jp.clip(data.qvel[self.dof+3:self.dof+6]*.1, -5, 5))
        return dict(state=actor, privileged_state=jp.concatenate([actor, data.actuator_force]))


def zero_unused_rows(params):
    result = copy.deepcopy(params)
    for index, name, dimension in ((1, "state", 123), (2, "privileged_state", 140)):
        np.testing.assert_array_equal(np.asarray(params[0].mean[name])[101:104], 0)
        kernel = np.asarray(result[index]["params"]["hidden_0"]["kernel"]).copy()
        if kernel.shape[0] != dimension:
            raise ValueError("Only the existing full-target single-network shape is supported")
        kernel[101:104] = 0
        result[index]["params"]["hidden_0"]["kernel"] = jp.asarray(kernel)
    return result


def convert(source, output):
    cfg = json.loads((source.parent.parent/"run_config.json").read_text())
    if cfg["method"] != "guided" or cfg["steps"] != 0 or cfg["env_config"].get("angular_velocity_observation"):
        raise ValueError("Start the controlled experiment from an original BC, not a PPO endpoint")
    if cfg["env_config"].get("observation_variant") != "initial_pose_mode_onehot_95_101":
        raise ValueError("Only mode-conditioned full-target BC has the reserved zero slots")
    before = pickle.loads(source.read_bytes())
    after = zero_unused_rows(before)
    runner = constrained.improvement.runner
    env = ModeConditionedGrasp(Path(cfg["input"])/"model.xml", "pure")
    net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
    rng = np.random.default_rng(19401)
    state = rng.normal(size=(64,123)).astype(np.float32)
    state[:,101:104] = 0
    old = dict(state=jp.asarray(state), privileged_state=jp.asarray(np.c_[state, rng.normal(size=(64,17))], jp.float32))
    new = {key:value.at[:,101:104].set(jp.asarray(rng.normal(size=(64,3)), jp.float32)) for key,value in old.items()}
    errors = {}
    for name, network, index in (("actor", net.policy_network, 1), ("critic", net.value_network, 2)):
        left = np.asarray(network.apply(before[0], before[index], old))
        right = np.asarray(network.apply(after[0], after[index], new))
        np.testing.assert_allclose(left, right, atol=1e-5, rtol=0)
        errors[name] = float(np.max(np.abs(left-right)))
    output.mkdir(parents=True, exist_ok=False)
    path = output/"params.pkl"
    path.write_bytes(pickle.dumps(jax.device_get(after)))
    cfg.update(output=str(output.resolve()), angular_velocity_ready=FEATURE,
        conversion_source_checkpoint=str(source.resolve()), conversion_source_sha256=digest(source))
    (output/"run_config.json").write_text(json.dumps(cfg, indent=2)+"\n")
    report = dict(status="converted_not_trained", source_checkpoint=str(source.resolve()),
        source_checkpoint_sha256=digest(source), derived_checkpoint_sha256=digest(path),
        actor_critic_zeroed_rows=[101,102,103], initial_function_errors=errors,
        normalizer_unchanged=True, additional_training_steps=0, source_sha256=digest(Path(__file__)))
    (output/"conversion.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--convert-source", type=Path)
    parser.add_argument("--conversion-output", type=Path)
    args, rest = parser.parse_known_args()
    if args.convert_source:
        if args.conversion_output is None or rest:
            parser.error("Conversion requires only source and output")
        convert(args.convert_source, args.conversion_output)
        return
    if rest[rest.index("--method")+1] != "guided":
        parser.error("Do not overwrite residual reference-action slots")
    checkpoint = Path(rest[rest.index("--checkpoint")+1])
    parent = checkpoint.parent
    if not (parent/"run_config.json").exists():
        parent = parent.parent
    cfg = json.loads((parent/"run_config.json").read_text())
    ready = cfg.get("angular_velocity_ready") == FEATURE
    resumed = cfg["env_config"].get("angular_velocity_observation") == FEATURE
    if not (ready or resumed):
        parser.error("Use a verified zero-input-row conversion or matching angular-feedback checkpoint")
    if ready:
        conversion = json.loads((parent/"conversion.json").read_text())
        assert digest(checkpoint) == conversion["derived_checkpoint_sha256"]
        params = pickle.loads(checkpoint.read_bytes())
        for index in (1,2):
            np.testing.assert_array_equal(np.asarray(params[index]["params"]["hidden_0"]["kernel"])[101:104], 0)
    original = constrained.improvement.ModeConditionedGrasp
    constrained.improvement.ModeConditionedGrasp = AngularFeedbackGrasp
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        constrained.improvement.ModeConditionedGrasp = original


if __name__ == "__main__":
    main()
