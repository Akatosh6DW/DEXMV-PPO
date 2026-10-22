"""Long replay of every queried recovery candidate, without further selection."""

import argparse
import json
from pathlib import Path
import pickle
import shutil

import jax
import jax.numpy as jp
import numpy as np

from probe_s09_recovery_teacher import ROOT, offset_controls, rollout, unpack_counters, objective_value
from scipy.optimize import differential_evolution
from retarget_s09_reference import digest
from run_s09_bc_conditioned import ModeConditionedGrasp
from run_s09_bc_phase_curriculum import KEEP
import run_s09_job as runner
from s09_suspension import SuspendedCylinder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refine-seconds", type=int, choices=(0,8), default=0,
                        help="Optional 8s teacher lookahead; never optimize on the full validation horizon")
    args = parser.parse_args()
    assert jax.default_backend() == "cpu"
    protocol = json.loads((args.probe/"protocol.json").read_text())
    assert protocol["failed_only"] and not protocol["smoke"]
    assert protocol["nominal_source"] == "phase_only_bc"
    checkpoint = Path(protocol["bc_checkpoint"])
    assert digest(checkpoint) == protocol["bc_checkpoint_sha256"]
    cfg = json.loads((checkpoint.parent.parent/"run_config.json").read_text())
    model = ROOT/"artifacts/orca_benchmark_suite/S09/inputs/model.xml"
    assert digest(model) == protocol["model_sha256"] == cfg["model_sha256"]
    params = pickle.loads(checkpoint.read_bytes())
    np.testing.assert_array_equal(np.asarray(params[1]["params"]["hidden_0"]["kernel"])[~KEEP], 0)
    env = ModeConditionedGrasp(model, "pure")
    net, _ = runner.initialize_matched(env, cfg["seed"], cfg["std"])
    policy = runner.networks.make_inference_fn(net)(params, deterministic=True)
    infer = jax.jit(lambda obs:policy(dict(state=obs), jax.random.PRNGKey(0))[0])
    reset = jax.jit(env.reset)
    cases = [r for r in json.loads((args.probe/"results.json").read_text()) if r["search_requested"]]
    assert cases and all(r["impulse"]=="none" and r["prefix_alive"] for r in cases)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output/"sources"
    sources.mkdir()
    for name in (Path(__file__).name, "probe_s09_recovery_teacher.py", "s09_suspension.py"):
        shutil.copyfile(ROOT/"scripts"/name, sources/name)
    (args.output/"protocol.json").write_text(json.dumps(dict(
        source_probe=str(args.probe.resolve()), source_protocol_sha256=digest(args.probe/"protocol.json"),
        source_results_sha256=digest(args.probe/"results.json"), checkpoint_sha256=digest(checkpoint),
        source_sha256=digest(Path(__file__)), seeds=[r["seed"] for r in cases],
        selection="all queried cases, including failed local optimizations", duration_sec=40,
        policy_optimizer_updates=0, refine_seconds=args.refine_seconds,
        search_iterations=3 if args.refine_seconds else 0,
        scope="training-state teacher validation, not trained BC/PPO or held-out evaluation"
    ), indent=2)+"\n")
    rows, physics_steps = [], 0
    for case in cases:
        seed = case["seed"]
        folder = args.output/f"seed{seed}"
        folder.mkdir()
        state = jax.device_get(reset(jax.random.PRNGKey(seed)))
        obs = np.tile(np.asarray(state.obs["state"]), (800,1))
        steps = jp.arange(800, dtype=jp.float32)
        obs[:,93] = np.asarray(jp.minimum(steps*.05/3.,1))
        obs[:,112] = np.asarray(steps/800)
        actions = np.asarray(infer(jp.asarray(obs)))
        sim = SuspendedCylinder(model.read_text())
        sim.reset(np.asarray(state.info["cube_spawn_pos"]), np.asarray(state.data.qpos[sim.oid+3:sim.oid+7]))
        with np.load(args.probe/f"seed{seed}_none/initial_state.npz") as z:
            snapshot = (z["state"].copy(), unpack_counters(z["spawn"], z["counters"]))
            delta, selected = z["selected_offset"].copy(), z["selected_actuators"].copy()
        low, high = sim.model.actuator_ctrlrange.astype(np.float32).T
        desired = low+(np.clip(actions,-1,1)+np.float32(1))*np.float32(.5)*(high-low)
        prefix = round(protocol["prefix_seconds"]/.05)
        if args.refine_seconds:
            history, best = [], [float("inf"), delta.copy()]
            horizon = args.refine_seconds*20
            def evaluate(change):
                preview, _ = rollout(sim, snapshot, offset_controls(
                    desired[prefix:prefix+horizon].astype(float), change, selected, sim.model.actuator_ctrlrange))
                cost = objective_value(preview, change)
                history.append(dict(evaluation=len(history), cost=cost,
                                    delta=np.asarray(change).tolist(), **preview))
                if cost < best[0]:
                    best[:] = [cost, np.array(change)]
                return cost
            evaluate(delta)
            evaluate(np.zeros(len(delta)))
            differential_evolution(evaluate, [(-.08,.08)]*len(selected), maxiter=3,
                popsize=4, seed=19201, polish=False, tol=0., x0=delta, workers=1)
            delta = best[1]
            (folder/"search.json").write_text(json.dumps(history, indent=2)+"\n")
        result = dict(seed=seed, mode=case["initial_mode"], selected_delta=delta.tolist())
        for name, change in (("unchanged_bc", np.zeros(len(selected))), ("selected_offset", delta)):
            targets = offset_controls(desired[prefix:].astype(float), change, selected, sim.model.actuator_ctrlrange)
            report, trace = rollout(sim, snapshot, targets)
            repeat, other = rollout(sim, snapshot, targets)
            assert report == repeat
            for key in trace:
                np.testing.assert_array_equal(trace[key], other[key])
            report["success30"] = bool(trace["success30"][-1] and trace["alive"][-1])
            report["final_time_sec"] = float(trace["time"][-1])
            report["best_hold_sec_including_prefix"] = float(trace["best_hold_sec"].max())
            np.savez_compressed(folder/f"{name}.npz", **trace)
            result[name] = report
        rows.append(result)
        physics_steps += sim.probe_control_steps
        (args.output/"results.json").write_text(json.dumps(rows, indent=2)+"\n")
        print(json.dumps(result), flush=True)
    summary = dict(status="completed", cases=len(rows),
        unchanged_successes=sum(r["unchanged_bc"]["success30"] for r in rows),
        corrected_successes=sum(r["selected_offset"]["success30"] for r in rows),
        total_native_control_steps=physics_steps, total_native_physics_substeps=physics_steps*sim.nsub,
        warning="Post-hoc teacher validation on queried training cases; not a student policy success rate")
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
