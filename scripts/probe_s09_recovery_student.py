"""Check recovery learning at a fixed teacher state, not held-out performance.

The requested prefix (three seconds by default) replays the verified teacher. Thereafter the
student supplies every action. This diagnostic splice is NOT a deployed policy.
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

import run_s09_job as runner
from orca_native_physics import STATE_SPEC
from retarget_s09_reference import digest
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_bc_phase_curriculum import KEEP
from s09_suspension import SuspendedCylinder


def feedback_clamp(obs):
    """Hold feedback at its initial value without removing time/mode inputs."""
    frozen = np.broadcast_to(obs[0], obs.shape).copy()
    frozen[:, KEEP] = obs[:, KEEP]
    return frozen


def load_cases(collector, ranked):
    """Only consume completed, audited ranked teachers in the new input mode."""
    if not ranked:
        path = collector/"collection.json"
        cases = [dict(r, trace_file=f"seed{r['seed']}_canonical_trace.npz")
                 for r in json.loads(path.read_text()) if r["qualified"]]
        return cases, path
    path = collector/"results.json"
    summary = json.loads((collector/"summary.json").read_text())
    proof = json.loads((collector/"report/evidence.json").read_text())
    assert summary["status"] == "completed" and not summary["smoke"]
    assert proof["status"] == "verified"
    assert proof["results_sha256"] == digest(path)
    assert proof["summary_sha256"] == digest(collector/"summary.json")
    cases = []
    for row in json.loads(path.read_text()):
        canonical = row["canonical"]
        if canonical and canonical["qualified"] and not row["short_success"]:
            trace_file = f"seed{row['seed']}/canonical_trace.npz"
            assert canonical["trace_sha256"] == proof["trace_hashes"][trace_file]
            cases.append(dict(seed=row["seed"], trace_file=trace_file,
                              trace_sha256=canonical["trace_sha256"], qualified=True))
    return cases, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-file", type=Path, default=Path("step_000000000/params.pkl"),
                        help="Explicit checkpoint path relative to each run")
    parser.add_argument("--ranked-teacher", action="store_true",
                        help="Read completed, verified full-horizon teacher ranking results")
    parser.add_argument("--teacher-forced", action="store_true",
                        help="Diagnostic only: replay predictions on recorded teacher observations")
    parser.add_argument("--prefix-steps", type=int, choices=(40, 60), default=60,
                        help="Teacher intervention step: 40 for 2s or original default 60 for 3s")
    args = parser.parse_args()
    if args.checkpoint_file.is_absolute() or ".." in args.checkpoint_file.parts:
        parser.error("Checkpoint must stay within the explicitly supplied run")
    assert jax.default_backend() == "cpu"
    jax.config.update("jax_default_matmul_precision", "highest")
    args.output.mkdir(parents=True, exist_ok=False)
    cases, collection_path = load_cases(args.collector, args.ranked_teacher)
    assert cases
    inference_factory = runner.networks.make_inference_fn
    results = []
    for run in args.runs:
        cfg = json.loads((run/"run_config.json").read_text())
        model = Path(cfg["input"])/"model.xml"
        assert cfg["method"] == "guided" and digest(model) == cfg["model_sha256"]
        assert cfg["env_config"]["observation_variant"] == "initial_pose_mode_onehot_95_101"
        env = ModeConditionedGrasp(model, "pure")
        net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
        checkpoint = run/args.checkpoint_file
        params = pickle.loads(checkpoint.read_bytes())
        policy = inference_factory(net)(params, deterministic=True)
        infer = jax.jit(lambda obs: policy(dict(state=obs), jax.random.PRNGKey(0))[0])
        for case in cases:
            seed = case["seed"]
            source = args.collector/case["trace_file"]
            assert digest(source) == case["trace_sha256"]
            with np.load(source) as z:
                teacher = dict(z)
            assert teacher["active"].all() and teacher["obs"].shape == (1, 800, 123)
            teacher_obs = teacher["obs"][0]
            predicted = np.asarray(infer(jp.asarray(teacher_obs)))
            clamped = np.asarray(infer(jp.asarray(feedback_clamp(teacher_obs))))
            targets = jp.asarray(teacher["action"][0])
            prefix = args.prefix_steps

            def mixed_policy(obs, key):
                index = jp.clip(jp.rint(obs["state"][..., 112]*800).astype(jp.int32), 0, 799)
                # Use the same per-step inference shape in both interventions;
                # dense offline prediction can alter GEMM rounding in contact.
                inputs = dict(state=jp.asarray(teacher_obs)[index]) if args.teacher_forced else obs
                learned = policy(inputs, key)[0]
                return jp.where((index < prefix)[..., None], targets[index], learned), {}

            runner.networks.make_inference_fn = lambda _: lambda _params, deterministic=True: mixed_policy
            try:
                evaluate = runner.evaluator(env, net)
                initial, trace = jax.device_get(evaluate(params, jp.array([seed], jp.int32)))
            finally:
                runner.networks.make_inference_fn = inference_factory
            np.testing.assert_array_equal(trace["action"][0, :prefix], teacher["action"][0, :prefix])
            np.testing.assert_allclose(trace["qpos"][0, :prefix], teacher["qpos"][0, :prefix], atol=3e-6, rtol=0)
            _, episodes = runner.summarize(trace, [seed])
            state = jax.device_get(jax.jit(env.reset)(jax.random.PRNGKey(seed)))
            sim = SuspendedCylinder(model.read_text())
            sim.reset(np.asarray(state.info["cube_spawn_pos"]), np.asarray(initial[0, sim.oid+3:sim.oid+7]))
            mujoco.mj_setState(sim.model, sim.data,
                np.asarray(state.data.integration_words).copy().view(np.float64), STATE_SPEC)
            max_q = max_v = 0.
            for step in np.flatnonzero(trace["active"][0]):
                row = sim.step(trace["desired_ctrl"][0, step])
                max_q = max(max_q, float(np.max(np.abs(sim.data.qpos-trace["qpos"][0, step]))))
                max_v = max(max_v, float(np.max(np.abs(sim.data.qvel-trace["qvel"][0, step]))))
            assert max_q < 3e-6 and max_v < 1e-4
            assert row["success30"] == episodes[0]["success30"]
            scale = np.diff(env.mj_model.actuator_ctrlrange, axis=1).ravel()/2
            error = (predicted-teacher["action"][0])*scale
            result = dict(run=run.name, checkpoint_sha256=digest(checkpoint),
                teacher_forced_observations=args.teacher_forced,
                teacher_trace_sha256=digest(source), prefix_control_steps=prefix,
                teacher_state_action_rmse_rad=float(np.sqrt(np.mean(error[prefix:]**2))),
                teacher_prefix_action_rmse_rad=float(np.sqrt(np.mean(error[:prefix]**2))),
                feedback_clamp_action_rmse_rad=float(np.sqrt(np.mean(((predicted-clamped)*scale)**2))),
                native_replay_max_qpos_error=max_q, native_replay_max_qvel_error=max_v,
                **episodes[0])
            path = args.output/f"{run.name}_seed{seed}.npz"
            np.savez_compressed(path, seeds=np.array([seed]), **trace)
            result["trace_sha256"] = digest(path)
            results.append(result)
            (args.output/"results.json").write_text(json.dumps(results, indent=2)+"\n")
            print(json.dumps(result), flush=True)
    shutil.copyfile(Path(__file__), args.output/Path(__file__).name)
    summary = dict(status="completed", cases=len(cases), policies=len(args.runs),
        prefix_control_steps=args.prefix_steps,
        teacher_forced_observations=args.teacher_forced,
        inference_shape="per-step policy inside scan in both modes; offline dense outputs used only for fit diagnostics",
        teacher_collection_sha256=digest(collection_path),
        ranked_teacher_input=args.ranked_teacher,
        checkpoint_relative_path=str(args.checkpoint_file),
        scope="Seen training-state diagnostic; shared teacher prefix then student control or oracle teacher-observation predictions. Not a deployable controller or held-out success rate.",
        source_sha256=digest(Path(__file__)))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
