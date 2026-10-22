"""One selective DAgger-inspired correction round on the current student's states.

Query the first six failed training trajectories with a live 2s prefix. A bounded
local optimizer supplies candidate teacher actions, validated through 40s and
the actual observation/action interface. Only teacher-controlled suffixes are
supervised. This is not full DAgger, MPC, new human video, or policy improvement.
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
from scipy.optimize import differential_evolution

from collect_s09_recovery_demo import merge_training
from probe_s09_recovery_teacher import ROOT, SPEC, COUNTERS, offset_controls, rollout, objective_value
from probe_s09_recovery_timing import phase_targets, fresh_prefix, verify_replay
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_bc_phase_curriculum import KEEP
from retarget_s09_reference import digest
from s09_suspension import SuspendedCylinder
import run_s09_job as runner


def select_queries(episodes, trace, limit=6):
    selected = []
    for index, row in enumerate(episodes):
        if (not row["success30"] and trace["active"][index, :40].all()
                and trace["alive"][index, :40].all() and len(selected) < limit):
            selected.append(index)
    return selected


def canonical_plan(env, net, seed, targets, folder, model):
    actions = env.ctrl_to_action(jp.asarray(targets, jp.float32))
    original = runner.networks.make_inference_fn

    def policy(obs, key):
        index = jp.clip(jp.rint(obs["state"][..., 112]*800).astype(jp.int32), 0, 799)
        return actions[index], {}

    runner.networks.make_inference_fn = lambda _: lambda _p, deterministic=True: policy
    try:
        _, trace = jax.device_get(runner.evaluator(env, net)((), jp.array([seed], jp.int32)))
    finally:
        runner.networks.make_inference_fn = original
    _, episodes = runner.summarize(trace, [seed])
    state = jax.device_get(jax.jit(env.reset)(jax.random.PRNGKey(seed)))
    sim = SuspendedCylinder(model.read_text())
    sim.reset(np.asarray(state.info["cube_spawn_pos"]), np.asarray(state.data.qpos[sim.oid+3:sim.oid+7]))
    mujoco.mj_setState(sim.model, sim.data, np.asarray(state.data.integration_words).copy().view(np.float64), SPEC)
    max_q = max_v = 0.
    for i in np.flatnonzero(trace["active"][0]):
        row = sim.step(trace["desired_ctrl"][0, i])
        max_q = max(max_q, float(np.max(np.abs(sim.data.qpos-trace["qpos"][0, i]))))
        max_v = max(max_v, float(np.max(np.abs(sim.data.qvel-trace["qvel"][0, i]))))
    assert max_q < 3e-6 and max_v < 1e-4 and row["success30"] == episodes[0]["success30"]
    path = folder/"canonical_trace.npz"
    np.savez_compressed(path, seeds=np.array([seed]), **trace)
    record = dict(**episodes[0], native_qpos_error=max_q, native_qvel_error=max_v,
                  trace_sha256=digest(path), qualified=episodes[0]["success30"])
    return record, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert jax.default_backend() == "cpu"
    jax.config.update("jax_default_matmul_precision", "highest")
    args.output.mkdir(parents=True, exist_ok=False)
    model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
    original_manifest = ROOT/"artifacts/orca_benchmark_suite/S09/demonstrations/dataset_manifest.json"
    source = json.loads(original_manifest.read_text())
    meta = json.loads((args.dataset.parent/"dataset_manifest.json").read_text())
    assert digest(args.dataset) == meta["dataset_sha256"]
    assert digest(model) == meta["model_sha256"] == source["model_sha256"]
    assert meta["reference_sha256"] == source["reference_sha256"]
    split = meta["train_episodes"]
    count = source.get("train_episodes", len(source["seeds"])*3//4)
    seeds = source["seeds"][:count]
    if args.smoke:
        seeds = seeds[:8]
    env = ModeConditionedGrasp(model, "pure")
    cfg = json.loads((args.student.parent.parent/"run_config.json").read_text())
    nominal_cfg = json.loads((args.nominal.parent.parent/"run_config.json").read_text())
    for c in (cfg, nominal_cfg):
        assert c["model_sha256"] == digest(model) and c["reference_sha256"] == meta["reference_sha256"]
        assert c["method"] == "guided" and c["env_config"]["observation_variant"] == "initial_pose_mode_onehot_95_101"
    assert set(seeds).issubset(meta["seeds"][:split])
    student = pickle.loads(args.student.read_bytes())
    nominal = pickle.loads(args.nominal.read_bytes())
    np.testing.assert_array_equal(np.asarray(nominal[1]["params"]["hidden_0"]["kernel"])[~KEEP], 0)
    net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
    nominal_policy = runner.networks.make_inference_fn(net)(nominal, deterministic=True)
    infer = jax.jit(lambda obs: nominal_policy(dict(state=obs), jax.random.PRNGKey(0))[0])
    protocol = dict(kind="selective_student_state_correction_round", student=str(args.student.resolve()),
        student_sha256=digest(args.student), nominal=str(args.nominal.resolve()), nominal_sha256=digest(args.nominal),
        dataset=str(args.dataset.resolve()), dataset_sha256=digest(args.dataset), model_sha256=digest(model),
        training_seeds=seeds, validation_or_test_used=False, prefix_steps=40,
        selection="first six failed full-episode student rollouts with a live 2s prefix in original training order",
        search_horizon_steps=4 if args.smoke else 160, search_iterations=0 if args.smoke else 3,
        search_seed=19201, offset_bound_rad=.08, popsize=4, policy_updates=0, smoke=args.smoke,
        source_sha256=digest(Path(__file__)),
        limitation="Selective correction round, not full DAgger. Teacher is a bounded optimizer, not a proven optimal expert.")
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    snapshots = args.output/"sources"
    snapshots.mkdir()
    for file in (Path(__file__), ROOT/"scripts/probe_s09_recovery_timing.py",
                 ROOT/"scripts/probe_s09_recovery_teacher.py", ROOT/"scripts/s09_suspension.py"):
        shutil.copyfile(file, snapshots/file.name)
    evaluate = runner.evaluator(env, net)
    batches = [jax.device_get(evaluate(student, jp.asarray(seeds[i:i+8], jp.int32)))
               for i in range(0, len(seeds), 8)]
    initial, trace = jax.tree_util.tree_map(lambda *values: np.concatenate(values), *batches)
    _, episodes = runner.summarize(trace, seeds)
    np.savez_compressed(args.output/"student_rollouts.npz", seeds=np.asarray(seeds), initial_qpos=initial, **trace)
    (args.output/"student_episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
    queried = select_queries(episodes, trace, limit=1 if args.smoke else 6)
    (args.output/"queries.json").write_text(json.dumps([seeds[i] for i in queried])+"\n")
    print(json.dumps(dict(stage="student_rollouts", cases=len(seeds),
        successes=sum(r["success30"] for r in episodes), queried=[seeds[i] for i in queried])), flush=True)
    records, added, search_steps, canonical_steps = [], [], 0, 0
    for index in queried:
        seed = seeds[index]
        folder = args.output/f"seed{seed}"
        folder.mkdir()
        state = jax.device_get(jax.jit(env.reset)(jax.random.PRNGKey(seed)))
        sim = SuspendedCylinder(model.read_text())
        desired = phase_targets(state, infer, sim.model.actuator_ctrlrange).astype(float)
        desired[:40] = trace["desired_ctrl"][index, :40]
        snapshot = fresh_prefix(sim, state, desired, 40)
        np.testing.assert_allclose(sim.data.qpos, trace["qpos"][index, 39], atol=3e-6, rtol=0)
        np.testing.assert_allclose(sim.data.qvel, trace["qvel"][index, 39], atol=1e-4, rtol=0)
        np.savez_compressed(folder/"initial_state.npz", state=snapshot[0], spawn=snapshot[1]["spawn"],
                            counters=np.array([snapshot[1][key] for key in COUNTERS[1:]]))
        selected = np.arange(7, 17)
        horizon = protocol["search_horizon_steps"]
        history, best = [], [float("inf"), np.zeros(10)]
        def objective(delta):
            result, _ = rollout(sim, snapshot, offset_controls(desired[40:40+horizon], delta,
                                                               selected, sim.model.actuator_ctrlrange))
            cost = objective_value(result, delta)
            history.append(dict(cost=cost, delta=np.asarray(delta).tolist(), **result))
            if cost < best[0]:
                best[:] = [cost, np.array(delta)]
            return cost
        objective(np.zeros(10))
        if not args.smoke:
            differential_evolution(objective, [(-.08, .08)]*10, maxiter=3, popsize=4,
                                   seed=19201, polish=False, tol=0., x0=np.zeros(10), workers=1)
        (folder/"search.json").write_text(json.dumps(history, indent=2)+"\n")
        result = dict(seed=seed, mode=episodes[index]["selected_reference"],
                      student_success30=episodes[index]["success30"], selected_delta=best[1].tolist())
        for label, delta in (("nominal_zero", np.zeros(10)), ("selected_offset", best[1])):
            targets = offset_controls(desired[40:], delta, selected, sim.model.actuator_ctrlrange)
            report, replay = verify_replay(sim, snapshot, targets)
            result[label] = report
            np.savez_compressed(folder/f"{label}.npz", **replay)
        search_steps += sim.probe_control_steps
        result["canonical"] = None
        if result["selected_offset"]["success30"]:
            desired[40:] = offset_controls(desired[40:], best[1], selected, sim.model.actuator_ctrlrange)
            record, collected = canonical_plan(env, net, seed, desired, folder, model)
            result["canonical"] = record
            canonical_steps += int(collected["active"].sum())
            if record["qualified"]:
                sample = {key: np.array(collected[key], copy=True) for key in ("obs", "action", "active")}
                sample["active"][:, :40] = False
                added.append(sample)
        records.append(result)
        (args.output/"collection.json").write_text(json.dumps(records, indent=2)+"\n")
        print(json.dumps(result), flush=True)
    if added:
        with np.load(args.dataset, allow_pickle=False) as z:
            old = {key: z[key] for key in ("obs", "action", "active")}
        new = {key: np.concatenate([item[key] for item in added]) for key in old}
        merged = merge_training(old, new, split)
        path = args.output/"trajectories.npz"
        np.savez_compressed(path, **merged)
        accepted = [r["seed"] for r in records if r["canonical"] and r["canonical"]["qualified"]]
        meta.update(train_episodes=split+len(added), dataset_sha256=digest(path),
            parent_dataset_sha256=digest(args.dataset), original_validation_preserved=True,
            seeds=meta["seeds"][:split]+accepted+meta["seeds"][split:],
            student_correction_episodes=len(added), additional_simulated_recovery_episodes=
                meta.get("additional_simulated_recovery_episodes", 0)+len(added),
            student_checkpoint_sha256=digest(args.student), student_prefix_supervised=False,
            new_human_videos=0, correction_protocol_sha256=digest(args.output/"protocol.json"))
        (args.output/"dataset_manifest.json").write_text(json.dumps(meta, indent=2)+"\n")
    (args.output/"summary.json").write_text(json.dumps(dict(status="completed", smoke=args.smoke,
        student_rollout_interactions=int(trace["active"].sum()), queried=len(queried),
        native_teacher_interactions=search_steps, canonical_collection_interactions=canonical_steps,
        canonical_validation_interactions=canonical_steps, accepted=len(added), policy_updates=0,
        limitation=protocol["limitation"]), indent=2)+"\n")


if __name__ == "__main__":
    main()
