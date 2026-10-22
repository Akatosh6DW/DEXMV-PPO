"""Validate teacher targets through the actual action/observation interface.

Only successful training-state teacher plans may augment the training split.
The original validation episodes are kept byte-for-byte unchanged. This is
demonstration collection, not a new learned controller or held-out evaluation.
"""

import argparse
import json
from pathlib import Path
import pickle
import shutil

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from probe_s09_recovery_teacher import ROOT, SPEC, offset_controls
from retarget_s09_reference import digest
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_bc_phase_curriculum import KEEP
import run_s09_job as runner
from s09_suspension import SuspendedCylinder


def merge_training(original, new, split):
    if not 0 < split < len(original["obs"]):
        raise ValueError("Original training and validation episodes are required")
    merged = {k:np.concatenate([original[k][:split], new[k], original[k][split:]])
              for k in ("obs", "action", "active")}
    for key in merged:
        np.testing.assert_array_equal(merged[key][split+len(new["obs"]):], original[key][split:])
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == "cpu"
    jax.config.update("jax_default_matmul_precision", "highest")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.teacher/"protocol.json").read_text())
    probe = Path(config["source_probe"])
    origin = json.loads((probe/"protocol.json").read_text())
    checkpoint = Path(origin["bc_checkpoint"])
    assert digest(checkpoint) == config["checkpoint_sha256"]
    params = pickle.loads(checkpoint.read_bytes())
    np.testing.assert_array_equal(np.asarray(params[1]["params"]["hidden_0"]["kernel"])[~KEEP], 0)
    model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
    assert digest(model) == origin["model_sha256"]
    source = ROOT/"artifacts/orca_benchmark_suite/S09/demonstrations"
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    assert digest(source/"trajectories.npz") == origin["dataset_sha256"] == manifest["dataset_sha256"]
    split = manifest.get("train_episodes", len(manifest["seeds"])*3//4)
    teacher_rows = json.loads((args.teacher/"results.json").read_text())
    cases = [r for r in teacher_rows if r["selected_offset"]["success30"]]
    assert cases and all(r["seed"] in manifest["seeds"][:split] for r in cases)
    env = ModeConditionedGrasp(model, "pure")
    cfg = json.loads((checkpoint.parent.parent/"run_config.json").read_text())
    net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
    original_inference = runner.networks.make_inference_fn
    policy = original_inference(net)(params, deterministic=True)
    infer = jax.jit(lambda obs:policy(dict(state=obs), jax.random.PRNGKey(0))[0])
    reset = jax.jit(env.reset)
    records, accepted = [], []
    for case in cases:
        seed = case["seed"]
        state = jax.device_get(reset(jax.random.PRNGKey(seed)))
        observations = np.tile(np.asarray(state.obs["state"]), (800,1))
        steps = jp.arange(800, dtype=jp.float32)
        observations[:,93] = np.asarray(jp.minimum(steps*.05/3.,1))
        observations[:,112] = np.asarray(steps/800)
        bc_actions = np.asarray(infer(jp.asarray(observations)))
        low, high = env.mj_model.actuator_ctrlrange.astype(np.float32).T
        targets = low+(np.clip(bc_actions,-1,1)+np.float32(1))*np.float32(.5)*(high-low)
        with np.load(probe/f"seed{seed}_none/initial_state.npz") as z:
            selected = z["selected_actuators"]
        prefix = round(origin["prefix_seconds"]/.05)
        targets = targets.astype(float)
        targets[prefix:] = offset_controls(targets[prefix:], np.array(case["selected_delta"]), selected,
                                          env.mj_model.actuator_ctrlrange)
        target_actions = env.ctrl_to_action(jp.asarray(targets, jp.float32))
        def action_policy(obs, key):
            index = jp.clip(jp.rint(obs["state"][...,112]*800).astype(jp.int32), 0, 799)
            return target_actions[index], {}
        runner.networks.make_inference_fn = lambda _:lambda _params, deterministic=True:action_policy
        try:
            evaluate = runner.evaluator(env, net)
            initial, trace = jax.device_get(evaluate((), jp.array([seed], jp.int32)))
        finally:
            runner.networks.make_inference_fn = original_inference
        _, episodes = runner.summarize(trace, [seed])
        sim = SuspendedCylinder(model.read_text())
        sim.reset(np.asarray(state.info["cube_spawn_pos"]), np.asarray(state.data.qpos[sim.oid+3:sim.oid+7]))
        mujoco.mj_setState(sim.model, sim.data, np.asarray(state.data.integration_words).copy().view(np.float64), SPEC)
        max_q = max_v = 0.
        for i in np.flatnonzero(trace["active"][0]):
            row = sim.step(trace["desired_ctrl"][0,i])
            max_q = max(max_q, float(np.max(np.abs(sim.data.qpos-trace["qpos"][0,i]))))
            max_v = max(max_v, float(np.max(np.abs(sim.data.qvel-trace["qvel"][0,i]))))
        assert max_q < 3e-6 and max_v < 1e-4 and row["success30"] == episodes[0]["success30"]
        path = args.output/f"seed{seed}_canonical_trace.npz"
        np.savez_compressed(path, seeds=np.array([seed]), **trace)
        record = dict(**episodes[0], native_replay_max_qpos_error=max_q,
            native_replay_max_qvel_error=max_v, trace_sha256=digest(path),
            teacher_selected_delta=case["selected_delta"], qualified=bool(episodes[0]["success30"]))
        records.append(record)
        if record["qualified"]:
            accepted.append({k:trace[k] for k in ("obs", "action", "active")})
        print(json.dumps(record), flush=True)
    (args.output/"collection.json").write_text(json.dumps(records, indent=2)+"\n")
    if accepted:
        with np.load(source/"trajectories.npz") as z:
            original = {k:z[k] for k in ("obs", "action", "active")}
        new = {k:np.concatenate([x[k] for x in accepted]) for k in original}
        merged = merge_training(original, new, split)
        np.savez_compressed(args.output/"trajectories.npz", **merged)
        manifest["original_coverage_rate"] = manifest.pop("coverage_rate", None)
        manifest["original_attempted_seeds"] = manifest.pop("attempted_seeds", [])
        manifest.update(train_episodes=split+len(accepted), dataset_sha256=digest(args.output/"trajectories.npz"),
            original_dataset_sha256=digest(source/"trajectories.npz"), original_validation_preserved=True,
            seeds=manifest["seeds"][:split]+[r["seed"] for r in records if r["qualified"]]+manifest["seeds"][split:],
            additional_simulated_recovery_episodes=len(accepted), new_human_videos=0,
            provenance="Original video-derived reference demonstrations plus physically optimized BC recovery training trajectories",
            teacher_protocol_sha256=digest(args.teacher/"protocol.json"), teacher_results_sha256=digest(args.teacher/"results.json"))
        (args.output/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    shutil.copyfile(Path(__file__), args.output/Path(__file__).name)
    (args.output/"summary.json").write_text(json.dumps(dict(status="completed", teacher_success_candidates=len(cases),
        canonical_successes=len(accepted), policy_updates=0, source_sha256=digest(Path(__file__)),
        warning="Training-only demonstration collection; a successful teacher is not a learned BC or PPO improvement"), indent=2)+"\n")


if __name__ == "__main__":
    main()
