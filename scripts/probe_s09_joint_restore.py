"""Inference-only causal probe: restore a joint group from the stage start.

This two-actor splice is NOT a trained BC+PPO or residual policy. Both source
checkpoints stay unchanged, and the original benchmark reports are untouched.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import pickle
import sys

import jax.numpy as jp
import mujoco
import numpy as np

import audit_s09_bc_repair as audit
import evaluate_s09_failure_quality as quality
from retarget_s09_reference import digest

runner = quality.quality.improvement.runner


def splice_actions(current, teacher, mask):
    if current.shape != teacher.shape or current.shape[-1] != 17 or mask.shape != (17,):
        raise ValueError("Use matching 17-joint actions and a 17-joint mask")
    return jp.where(mask, teacher, current)


@contextmanager
def restored_joints(teacher, mask):
    original = runner.networks.make_inference_fn

    def factory(net, *args, **kwargs):
        make = original(net, *args, **kwargs)

        def build(params, deterministic=True):
            if not deterministic:
                raise ValueError("Joint restoration is deterministic evaluation only")
            current_policy = make(params, deterministic=True)
            teacher_policy = make(teacher, deterministic=True)

            def policy(obs, key):
                current, _ = current_policy(obs, key)
                source, _ = teacher_policy(obs, key)
                return splice_actions(current, source, mask), {}

            return policy

        return build

    runner.networks.make_inference_fn = factory
    try:
        yield
    finally:
        runner.networks.make_inference_fn = original


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--restore-joints", choices=("thumb", "non_thumb"))
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--audit-run", type=Path)
    args, rest = parser.parse_known_args()
    if args.audit_run:
        if rest or args.restore_joints or args.teacher_checkpoint:
            parser.error("Audit settings are read only from the saved probe metadata")
        output = args.audit_run
        metadata = json.loads((output/"joint_restore_config.json").read_text())
        current_cfg = json.loads((output/"run_config.json").read_text())
        checkpoint = Path(metadata["current_checkpoint"])
        teacher_path = Path(metadata["teacher_checkpoint"])
        group = metadata["restored_group"]
        model_path = Path(current_cfg["input"])/"model.xml"
    else:
        if not args.restore_joints or not args.teacher_checkpoint:
            parser.error("Specify the restored group and teacher checkpoint")
        if rest[rest.index("--steps")+1] != "0" or rest[rest.index("--method")+1] != "guided":
            parser.error("Only zero-update full-target diagnostics are supported")
        checkpoint = Path(rest[rest.index("--checkpoint")+1])
        teacher_path, group = args.teacher_checkpoint, args.restore_joints
        current_cfg = json.loads((checkpoint.parent.parent/"run_config.json").read_text())
        output = Path(rest[rest.index("--output")+1])
        model_path = Path(rest[rest.index("--input")+1])/"model.xml"
    teacher_cfg = json.loads((teacher_path.parent.parent/"run_config.json").read_text())
    for key in ("model_sha256", "reference_sha256", "env_config", "method", "actor_dim"):
        assert current_cfg[key] == teacher_cfg[key], key
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert digest(model_path) == current_cfg["model_sha256"] and model.nu == 17
    names = [model.actuator(i).name for i in range(model.nu)]
    thumb = np.array([name.startswith("right_t-") for name in names])
    assert thumb.sum() == 4
    mask = thumb if group == "thumb" else ~thumb
    current_hash, teacher_hash = digest(checkpoint), digest(teacher_path)
    teacher = pickle.loads(teacher_path.read_bytes())
    expected = dict(current_checkpoint=str(checkpoint.resolve()), current_checkpoint_sha256=current_hash,
        teacher_checkpoint=str(teacher_path.resolve()), teacher_checkpoint_sha256=teacher_hash,
        restored_group=group, restored_actuators=[name for name,use in zip(names,mask) if use],
        optimizer_updates=0, architecture="diagnostic_two_actor_joint_splice_not_trained_policy",
        warning="Non-thumb group includes the wrist; not a final benchmark method or residual addition")
    if args.audit_run:
        for key, value in expected.items():
            assert metadata[key] == value, key
        sys.argv = [sys.argv[0], "--run", str(output), "--step", "0"]
        with restored_joints(teacher, jp.asarray(mask)):
            audit.main()
        path = output/"step_000000000/native_replay_audit.json"
        result = json.loads(path.read_text())
        result.update(control_architecture=expected["architecture"],
            joint_restore_config_sha256=digest(output/"joint_restore_config.json"),
            teacher_checkpoint_sha256=teacher_hash)
        path.write_text(json.dumps(result, indent=2)+"\n")
    else:
        sys.argv = [sys.argv[0], *rest]
        with restored_joints(teacher, jp.asarray(mask)):
            quality.main()
        expected["source_sha256"] = digest(output/"sources"/Path(__file__).name)
        (output/"joint_restore_config.json").write_text(json.dumps(expected, indent=2)+"\n")
        path = output/"run_config.json"
        cfg = json.loads(path.read_text())
        cfg["policy_architecture"] = expected["architecture"]
        path.write_text(json.dumps(cfg, indent=2)+"\n")
    assert digest(checkpoint) == current_hash and digest(teacher_path) == teacher_hash


if __name__ == "__main__":
    main()
