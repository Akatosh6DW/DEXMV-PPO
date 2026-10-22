"""Verify a completed warmup's parameter boundaries and optional calibration.

The optional diagnostic dataset is never used by PPO. Calibration is compared
only before and immediately after the actor-frozen period, not after the policy
has changed. Saved-preview physics audits remain a separate requirement.
"""

import argparse
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np

from diagnose_s09_learned_critic import calibration
from retarget_s09_reference import digest
import run_s09_learned_base_residual as learned
from run_s09_learned_residual_experts import residual_expert_factory


def difference(before, after):
    a, ta = jax.tree_util.tree_flatten(before)
    b, tb = jax.tree_util.tree_flatten(after)
    if ta != tb:
        raise ValueError("Parameter structures differ")
    changed, maximum = 0, 0.
    for x, y in zip(a, b):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError("Parameter shape or dtype changed")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite parameter")
        changed += int(np.count_nonzero(x != y))
        maximum = max(maximum, float(np.max(np.abs(x-y), initial=0)))
    return dict(changed_elements=changed, maximum_absolute_difference=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    cfg = read(args.run/"run_config.json")
    meta = read(args.run/"residual_warmup_config.json")
    summary = read(args.run/"summary.json")
    warm, final = meta["actor_warmup_transitions"], summary["final_step"]
    assert summary["status"] == "completed" and 0 < warm < final
    assert meta["actor_warmup_gradient_updates"] == warm//2560*16
    assert meta["total_online_transitions"] == cfg["steps"] == final
    for source, field in (("run_s09_learned_residual_warmup.py", "source_sha256"),
                          ("run_s09_constrained_ppo.py", "optimizer_source_sha256"),
                          ("run_s09_critic_warmup.py", "delay_source_sha256")):
        assert digest(args.run/"sources"/source) == meta[field] == cfg["sources"][source]
    params, evidence = {}, {}
    for step in (0, warm, final):
        folder = args.run/f"step_{step:09d}"
        audit = read(folder/"native_replay_audit.json")
        assert audit["status"] == "verified"
        assert audit["checkpoint_sha256"] == digest(folder/"params.pkl")
        assert audit["trace_sha256"] == digest(folder/"preview_trajectories.npz")
        params[step] = pickle.loads((folder/"params.pkl").read_bytes())
        evidence[str(step)] = dict(checkpoint_sha256=audit["checkpoint_sha256"],
                                  audit_sha256=digest(folder/"native_replay_audit.json"))
    changes = dict(actor_during_warmup=difference(params[0][1], params[warm][1]),
                   critic_during_warmup=difference(params[0][2], params[warm][2]),
                   actor_after_warmup=difference(params[warm][1], params[final][1]))
    assert changes["actor_during_warmup"]["changed_elements"] == 0
    assert changes["critic_during_warmup"]["changed_elements"] > 0
    assert changes["actor_after_warmup"]["changed_elements"] > 0
    rows = []
    extra = None
    if args.calibration:
        diagnostic = read(args.calibration/"summary.json")
        assert diagnostic["status"] == "completed"
        assert diagnostic["checkpoint_sha256"] == evidence["0"]["checkpoint_sha256"]
        path = args.calibration/"critic_calibration.npz"
        assert diagnostic["data_sha256"] == digest(path)
        protocol = read(args.run/"learned_base_config.json")
        assert diagnostic["base_sha256"] == protocol["base_sha256"]
        action, _, _ = learned.load_base(Path(protocol["base_checkpoint"]), Path(cfg["input"])/"model.xml")
        net = residual_expert_factory(action)(dict(state=(123,), privileged_state=(140,)), 17)
        with np.load(path, allow_pickle=False) as z:
            obs, target, active = z["privileged_obs"], z["returns"], z["active"]
            mode = z["mode"]
        jax.config.update("jax_default_matmul_precision", "highest")
        predict = jax.jit(lambda stat, val, x: net.value_network.apply(stat, val,
            dict(state=x[...,:123], privileged_state=x)))
        frames = np.arange(800)[None,:]
        for step in (0, warm):
            prediction = np.asarray(predict(params[step][0], params[step][2], jp.asarray(obs)))
            for group in range(-1, 6):
                cases = np.ones(len(obs), bool) if group == -1 else mode == group
                for phase, times in (("all", frames>=0), ("acquisition", frames<60),
                                     ("released_before_30s", (frames>=60)&(frames<600))):
                    rows.append(dict(step=step, mode=group, phase=phase,
                        **calibration(prediction, target, active & cases[:,None] & times)))
        extra = dict(dataset_sha256=digest(path), diagnostic_interactions=diagnostic["actual_environment_interactions"],
                     used_for_training=False, caution="Correlated diagnostic frames, not independent test episodes")
    result = dict(status="verified", changes=changes, evidence=evidence, calibration=rows,
        diagnostic=extra, warmup_steps=warm, total_steps=final, actor_active_steps=final-warm,
        source_sha256=digest(Path(__file__)), native_audit_scope="Saved previews only",
        moment_freeze_evidence="Delayed optimizer source plus unit tests; optimizer states are not checkpointed")
    (args.run/"warmup_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
