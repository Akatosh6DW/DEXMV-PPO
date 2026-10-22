"""Test a fixed learned recovery teacher near two predeclared training states.

Uses D44's audited 2-second prefix, then the already fixed PPO portfolio.
No new policy updates or expert selection. Nominal replay is diagnostic only.
"""

import argparse
import json
from pathlib import Path
import pickle
import shutil

import jax
import jax.numpy as jp
import numpy as np

from probe_s09_feedback_perturbation import (
    BASE, ROOT, compare_trace, impulse_state, make_rollout, native_audit,
)
from probe_s09_recovery_student import load_cases
from retarget_s09_reference import digest
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_mode_experts import ARCHITECTURE, expert_networks
import run_s09_job as runner


def prefix_factory(actions):
    actions = jp.asarray(actions)
    if actions.shape != (800, 17):
        raise ValueError("Expected one complete teacher action trace")

    def factory(params, deterministic=True):
        def policy(obs, key):
            index = jp.clip(jp.rint(obs["state"][..., 112] * 800).astype(jp.int32), 0, 799)
            return actions[index], {}
        return policy
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--intervention-step", type=int, choices=(40, 80), default=40,
                        help="Object velocity intervention at 2s or 4s; teacher prefix remains 2s")
    args = parser.parse_args()
    assert jax.default_backend() == "cpu"
    jax.config.update("jax_default_matmul_precision", "highest")
    collector = BASE / "D44_teacher_full_pool"
    run = BASE / "confirmation_portfolio_20261019/candidate"
    checkpoint = run / "params.pkl"
    cfg = json.loads((run / "run_config.json").read_text())
    assert cfg["policy_architecture"] == ARCHITECTURE and cfg["method"] == "guided"
    model = ROOT / "artifacts/orca_benchmark_suite/S09/inputs/model.xml"
    assert digest(model) == cfg["model_sha256"]
    assert digest(model.parent / "reference.npz") == cfg["reference_sha256"]
    cases, collection_path = load_cases(collector, True)
    assert [r["seed"] for r in cases] == [90531, 90540]
    manifest = json.loads((model.parent.parent / "demonstrations/dataset_manifest.json").read_text())
    split = manifest.get("train_episodes", len(manifest["seeds"]) * 3 // 4)
    assert {r["seed"] for r in cases}.issubset(manifest["seeds"][:split])
    impulses = [("none", [0., 0., 0.])]
    for axis, name in enumerate("xyz"):
        for sign, label in ((1, "pos"), (-1, "neg")):
            delta = [0., 0., 0.]
            delta[axis] = sign * .02
            impulses.append((f"{name}_{label}", delta))
    steps = 800 - args.intervention_step
    if args.smoke:
        cases, impulses, steps = cases[:1], impulses[:2], 80
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output / "sources"
    sources.mkdir()
    names = (Path(__file__).name, "probe_s09_feedback_perturbation.py", "probe_s09_recovery_student.py",
             "run_s09_mode_experts.py", "orca_s09_env.py", "orca_native_physics.py", "s09_suspension.py")
    for name in names:
        shutil.copyfile(ROOT / "scripts" / name, sources / name)
    protocol = dict(seeds=[r["seed"] for r in cases], impulses=impulses, impulse_unit="m/s",
        prefix_steps=40, intervention_step=args.intervention_step, suffix_steps=steps,
        additional_policy_updates=0, inherited_online_steps=1310720,
        teacher_checkpoint=str(checkpoint.relative_to(ROOT)), teacher_checkpoint_sha256=digest(checkpoint),
        collection_sha256=digest(collection_path), model_sha256=digest(model),
        reference_sha256=digest(model.parent / "reference.npz"), smoke=args.smoke,
        execution_sources={name: digest(sources / name) for name in names},
        scope="Two predeclared training cases; existing learned teacher with historical dev-based expert selection; not new video or heldout",
        replay_after_nominal_death="Frozen terminal observation; exclude those times from feedback causality")
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    params = pickle.loads(checkpoint.read_bytes())
    rows = []
    with expert_networks():
        env = ModeConditionedGrasp(model, "pure")
        net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
        evaluate = make_rollout(env, net, steps)
        reset = jax.jit(env.reset)
        for case in cases:
            trace_path = collector / case["trace_file"]
            assert digest(trace_path) == case["trace_sha256"]
            with np.load(trace_path) as data:
                teacher = dict(data)
            rollout = make_rollout(env, net, 40, prefix_factory(teacher["action"][0]))
            start = reset(jax.random.PRNGKey(case["seed"]))
            state, prefix = jax.device_get(rollout(params, start, jp.zeros((40, 123)), jp.bool_(False)))
            assert int(state.info["step"]) == 40 and prefix["alive"].all()
            np.testing.assert_array_equal(prefix["action"], teacher["action"][0, :40])
            np.testing.assert_allclose(prefix["qpos"], teacher["qpos"][0, :40], atol=3e-6, rtol=0)
            native_audit(model, start, prefix)
            np.savez_compressed(args.output / f"prefix_seed{case['seed']}.npz", **prefix)
            if args.intervention_step > 40:
                warmup_steps = args.intervention_step - 40
                warmup = make_rollout(env, net, warmup_steps)
                warm_state, trace = jax.device_get(warmup(params, state, jp.zeros((warmup_steps, 123)), jp.bool_(False)))
                # Both predeclared cases must reach the intervention; do not
                # silently remove an early failure from the denominator.
                assert trace["alive"].all() and int(warm_state.info["step"]) == args.intervention_step
                native_audit(model, state, trace)
                np.savez_compressed(args.output / f"warmup_seed{case['seed']}.npz", **trace)
                state = warm_state
            _, nominal = jax.device_get(evaluate(params, state, jp.zeros((steps, 123)), jp.bool_(False)))
            for label, delta in impulses:
                changed = impulse_state(env, state, delta)
                for replay in (False, True):
                    if label == "none" and not replay:
                        trace = nominal
                    else:
                        _, trace = jax.device_get(evaluate(params, changed, jp.asarray(nominal["obs"]), jp.bool_(replay)))
                    if label == "none":
                        for key in trace:
                            np.testing.assert_array_equal(trace[key], nominal[key])
                    audit = native_audit(model, changed, trace)
                    _, episodes = runner.summarize({k: v[None] for k, v in trace.items()}, [case["seed"]])
                    filename = f"seed{case['seed']}_{label}_{'replay' if replay else 'feedback'}.npz"
                    np.savez_compressed(args.output / filename, **trace)
                    row = dict(run="fixed_portfolio", impulse=label, replay=replay,
                        nominal_alive_at_end=bool(nominal["alive"][-1]),
                        nominal_duration_sec=float(nominal["active"].sum() * .05),
                        checkpoint_sha256=digest(checkpoint), trace_file=filename,
                        trace_sha256=digest(args.output / filename), **episodes[0], **audit,
                        **compare_trace(trace, nominal, env.oid, env.dof))
                    rows.append(row)
                    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
    summary = dict(status="completed", smoke=args.smoke, cases=len(rows), policy_updates=0,
        control_steps=sum(r["native_steps"] for r in rows),
        native_validation_steps=sum(r["native_steps"] for r in rows),
        prefix_control_steps=40 * len(cases),
        learned_warmup_control_steps=(args.intervention_step - 40) * len(cases),
        protocol_sha256=digest(args.output / "protocol.json"),
        results_sha256=digest(args.output / "results.json"))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
