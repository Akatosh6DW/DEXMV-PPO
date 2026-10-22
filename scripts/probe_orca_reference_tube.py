"""Project a fixed absolute PPO controller into video-centered tubes: diagnostic only."""

import argparse
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jp
import numpy as np

from audit_orca_trace_geometry import audit
from evaluate_orca_matched_native import actor_control, rollout
from orca_hic53_env import check_hic53_gates
from orca_matched_env import OrcaMatchedGrasp
from orca_residual_controls import load_reference
from orca_residual_metrics import summarize
from prepare_hic_orca_object import digest
from train_orca_matched import initialize_matched


def correction_metrics(trace):
    return dict(requested_residual_max_rad=float(np.abs(trace["requested_residual"]).max()),
                effective_residual_max_rad=float(np.abs(trace["effective_residual"]).max()))


def projected_actor(env, net, reference, bound):
    original = actor_control(env, net)
    reference = jp.asarray(reference)
    @jax.jit
    def run(params, qpos, qvel, ctrl, force, geoms, xpos, info):
        obs, action, _, _, _, desired, _ = original(params, qpos, qvel, ctrl, force, geoms, xpos, info)
        base = reference[jp.minimum(info["step"], len(reference) - 1)]
        residual = jp.clip(desired - base, -bound, bound)
        desired = jp.clip(base + residual, env._lowers, env._uppers)
        absolute = env.ctrl_to_action(desired)
        target = env._lowers + (absolute + 1) * .5 * (env._uppers - env._lowers)
        applied = jp.clip(target, ctrl - .1, ctrl + .1)
        return obs, action, absolute, base, residual, desired, applied
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--rescore", action="store_true", help="Recompute correction metrics from already saved diagnostic trajectories")
    args = p.parse_args()
    if jax.default_backend() != "cpu" or args.count < 1:
        p.error("Positive episode count and JAX_PLATFORMS=cpu required")
    config = json.loads((args.run / "run_config.json").read_text())
    if config["method"] != "ppo":
        p.error("This diagnostic expects an absolute PPO controller")
    model, ref, gate = [Path(config[k]) for k in ("model", "reference", "gate")]
    check_hic53_gates(model, ref, gate)
    assert digest(model) == config["model_sha256"] and digest(ref) == config["reference_sha256"]
    env = OrcaMatchedGrasp(model, "pure", ref, config["weights"], tactile=config["tactile"] == "on")
    reference, _ = load_reference(ref, env._lowers, env._uppers, env._config.initial_ctrl, .05, 800, .1)
    net, _ = initialize_matched(env, config["seed"], config["init_std"])
    checkpoint = args.run / f"step_{args.step:09d}/params.pkl"
    params = jax.tree.map(np.asarray, pickle.loads(checkpoint.read_bytes()))
    if args.rescore:
        path = args.output / "report.json"
        report = json.loads(path.read_text())
        assert report["checkpoint_sha256"] == digest(checkpoint)
        for row in report["records"]:
            with np.load(args.output / f"bound_{row['bound_rad']:.2f}.npz") as trace:
                row.update(correction_metrics(trace))
                assert row["effective_residual_max_rad"] <= row["bound_rad"] + 1e-5
        report["correction_metric_audit"] = "Recomputed from saved arrays, not pure-controller placeholder metrics"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("RESCORED", len(report["records"]), "bounds")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    seeds = list(range(2100, 2100 + args.count))
    for bound in (0., .04, .08, .12, .15, .2, .25):
        inference = projected_actor(env, net, reference, bound)
        episodes = [rollout(env, inference, params, seed) for seed in seeds]
        trace = {k: np.stack([e[k] for e in episodes]) for k in episodes[0]}
        trace["effective_residual"] = trace["ctrl"] - trace["video_base"]
        assert np.max(np.abs(trace["effective_residual"])) <= bound + 1e-5
        row, holds, _ = summarize(trace, "pure", .08, env._lowers, env._uppers)
        row.update(correction_metrics(trace))
        final_valid = trace["alive"][:, -1] & (trace["hold"][:, -1] > 0)
        row.update(bound_rad=bound, success15=float(np.mean((holds >= 15) & final_valid)),
                   success15_any=float(np.mean(holds >= 15)),
                   max_force_n=float(trace["finger_normal_force"][trace["active"]].max()))
        path = args.output / f"bound_{bound:.2f}.npz"
        np.savez_compressed(path, **trace)
        geometry = audit(model, path)
        row["max_penetration_mm"] = geometry["max_penetration_m"] * 1000
        row["holds"] = holds.tolist()
        records.append(row)
        report = dict(kind="Fixed absolute PPO projected into reference tube; NOT residual-policy training",
                      reference_sha256=digest(ref), model_sha256=digest(model), checkpoint_sha256=digest(checkpoint),
                      seeds=seeds, records=records, no_source_reference_modified=True,
                      actor_observation="Original absolute-PPO observation; reference only used by projection",
                      note="A failed projection does not prove that no trained residual policy within that bound can succeed")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("TUBE", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
