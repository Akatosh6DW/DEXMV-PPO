# 中文阅读副本：18 S04正式回放评估入口
# 原始入口：scripts/run_orca_s03_eval.py
# 仅用于阅读，保留原路径假设；不要从此副本启动训练。运行请使用原始入口。
# 本副本只增加注释，生成时核验AST与非注释词法序列完全一致。
"""Audited held-out evaluations, including matched fixed residual replay."""

import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import jax
import numpy as np
from orca_s03_env import OrcaS03Grasp
from orca_s03_objects import OrcaS03ObjectGrasp
from orca_s03_evaluation import evaluator, recorded_actions, assert_matched_prefix
from prepare_hic_orca_object import digest
from run_orca_suite_job import MODEL, REFERENCE, summarize
from train_orca_acquisition import initial_gate
from train_orca_matched import initialize_matched


# 导读：评估只加载权重不训练；zero无需权重，其他路线必须提供匹配检查点。
def validate_job(job):
    if job["method"] not in ("ppo", "guided", "residual", "bc", "zero"):
        raise ValueError("Unknown evaluation method")
    if job.get("steps", 0) or job["eval_count"] < 1:
        raise ValueError("Evaluation does not train")
    if bool(job.get("checkpoint")) != (job["method"] != "zero"):
        raise ValueError("Exactly the learned methods require a checkpoint")
    if job.get("paired_replay") and (job["method"] != "residual" or job.get("shape")):
        raise ValueError("Replay ablation requires the original residual task")
    if job.get("paired_replay") and not 0 < job.get("push_force", 0) <= 2:
        raise ValueError("Replay ablation requires a nonzero, bounded force")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    validate_job(job)
    out = ROOT / job["output"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    jax.config.update("jax_pmap_shmap_merge", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "artifacts/orca30/jax_cache"))
    if jax.default_backend() != "gpu":
        raise RuntimeError("Expected aero_rl GPU backend")
    arm = "residual" if job["method"] in ("zero", "residual") else "pure"
    settings = dict(benchmark=job.get("benchmark", "B0_nominal"), audit=True,
                    shaping=job.get("shaping", 1.), residual_gain=0. if job["method"] == "zero" else 1.,
                    push_force=job.get("push_force"))
    if job.get("paired_replay"):
        settings["push_force"] = 0.
    if job.get("shape"):
        env = OrcaS03ObjectGrasp(MODEL, arm, REFERENCE, shape=job["shape"], **settings)
        env.export_model(out / "evaluation_model.xml")
        evaluated_model = out / "evaluation_model.xml"
    else:
        env = OrcaS03Grasp(MODEL, arm, REFERENCE, **settings)
        evaluated_model = MODEL
    gate = initial_gate(env)
    net, params = initialize_matched(env, job["seed"], job["std"])
    meta = dict(job, job_sha256=digest(args.job), arm=arm, model=str(evaluated_model),
                model_sha256=digest(evaluated_model), training_model_sha256=digest(MODEL),
                reference=str(REFERENCE), reference_sha256=digest(REFERENCE),
                actor_dim=123, critic_dim=140, initial_gate=gate, audit=True,
                object_transfer=bool(job.get("shape")), source_sha256={})
    if job.get("checkpoint"):
        checkpoint = ROOT / job["checkpoint"]
        config = checkpoint.parent / "run_config.json"
        if not config.exists():
            config = checkpoint.parent.parent / "run_config.json"
        prior = json.loads(config.read_text())
        for key, expected in dict(model_sha256=digest(MODEL), reference_sha256=digest(REFERENCE),
                                  arm=arm, actor_dim=123, critic_dim=140).items():
            if prior[key] != expected:
                raise ValueError("Checkpoint source contract mismatch: " + key)
        if prior["seed"] != job["seed"]:
            raise ValueError("Checkpoint seed mismatch")
        params = pickle.loads(checkpoint.read_bytes())
        meta["checkpoint_sha256"] = digest(checkpoint)
        meta["training_config_sha256"] = digest(config)
    sources = out / "sources"
    sources.mkdir()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().parent == ROOT / "scripts" and filename.endswith(".py"):
            path = Path(filename).resolve()
            shutil.copyfile(path, sources / path.name)
            meta["source_sha256"][path.name] = digest(path)
    (out / "run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    seeds = list(range(job["eval_start"], job["eval_start"] + job["eval_count"]))
    blank = np.zeros((len(seeds), 800, 17), np.float32)
    params = jax.tree_util.tree_map(np.asarray, jax.device_get(params))
    summaries = {}

    # 导读：记录全部回合指标，仅保存前4个可视轨迹；视频数不等于统计样本数。
    def save(name, environment, result):
        initial, trace = result
        if not np.isfinite(initial).all() or not all(np.isfinite(v).all() for v in trace.values()):
            raise FloatingPointError("Nonfinite evaluation")
        row, episodes = summarize(trace, environment, seeds)
        for e in episodes:
            e["full_episode_clean_success30"] = bool(e["clean_success30"] and e["max_penetration_m"] <= .003)
        directory = out / name
        directory.mkdir()
        (directory / "episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        np.savez_compressed(directory / "initial_states.npz", qpos=initial, seeds=seeds)
        np.savez_compressed(directory / "preview_trajectories.npz", **{k: v[:4] for k, v in trace.items()})
        summaries[name] = row
        print(json.dumps(dict(mode=name, **row)), flush=True)

    initial, nominal = jax.device_get(evaluator(env, net, seeds)(params, blank))
    if not job.get("paired_replay"):
        save("evaluation", env, (initial, nominal))
    else:
        actions = recorded_actions(nominal)
        np.savez_compressed(out / "nominal_recorded_actions.npz", seeds=seeds, actions=actions,
                            active=nominal["active"])
        save("nominal_closed", env, (initial, nominal))
        replay_initial, replay = jax.device_get(evaluator(env, net, seeds, replay=True)(params, actions))
        np.testing.assert_array_equal(initial, replay_initial)
        assert_matched_prefix(nominal, replay)
        save("nominal_replay", env, (replay_initial, replay))
        del replay
        forced = OrcaS03Grasp(MODEL, arm, REFERENCE, **dict(settings, push_force=job["push_force"]))
        forced_initial, closed = jax.device_get(evaluator(forced, net, seeds)(params, blank))
        np.testing.assert_array_equal(initial, forced_initial)
        assert_matched_prefix(nominal, closed, length=200)
        save("pushed_closed", forced, (forced_initial, closed))
        _, fixed = jax.device_get(evaluator(forced, net, seeds, replay=True)(params, actions))
        assert_matched_prefix(nominal, fixed, length=200)
        np.testing.assert_array_equal(actions, fixed["policy_action"])
        save("pushed_replay", forced, (forced_initial, fixed))
        rows = []
        for i, seed in enumerate(seeds):
            common = nominal["active"][i] & closed["active"][i] & fixed["active"][i]
            common[:200] = False
            change = closed["requested_residual"][i] - fixed["requested_residual"][i]
            rows.append(dict(seed=seed, common_post_push_steps=int(common.sum()),
                             feedback_change_rms_rad=float(np.sqrt(np.mean(change[common]**2))) if common.any() else None,
                             recorded_source_active_steps=int(nominal["active"][i].sum())))
        (out / "feedback_diagnostic.json").write_text(json.dumps(rows, indent=2)+"\n")
        meta["replay_audit"] = dict(nominal_active_trajectory_exact=True,
                                     pre_push_active_trajectory_exact=True,
                                     replay_actions_exact=True, post_source_terminal="zero residual")
        (out / "run_config.json").write_text(json.dumps(meta, indent=2)+"\n")
    summary = dict(status="completed", job_sha256=digest(args.job), summaries=summaries,
                   elapsed_sec=time.monotonic()-started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
