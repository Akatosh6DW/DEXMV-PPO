"""Bounded single-joint endpoint search; simulator calibration, not video evidence."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import jax
import numpy as np

from orca_shape_pilot_env import OrcaShapePilot
from orca_s03_evaluation import evaluator
from prepare_hic_orca_object import digest
from run_orca_suite_job import MODEL, REFERENCE, summarize
from train_orca_acquisition import initial_gate
from train_orca_matched import initialize_matched


def ramp(times):
    return np.clip((np.asarray(times) - 1.5) / 1.5, 0., 1.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-offset", type=Path)
    p.add_argument("--count", type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_default_matmul_precision", "highest")
    started = time.monotonic()
    env = OrcaShapePilot(MODEL, "residual", REFERENCE, shape="cylinder", audit=True)
    gate = initial_gate(env)
    net, params = initialize_matched(env, 32000, .7)
    seeds = list(range(94000, 94000 + args.count))
    evaluate = evaluator(env, net, seeds, replay=True)
    base = np.zeros(17)
    if args.base_offset:
        base = np.asarray(json.loads(args.base_offset.read_text())["offset_rad"])
    candidates = [("baseline", base)]
    for joint in (10, 11, 12, 13, 14, 15, 16):
        for delta in (-.08, -.04, .04, .08):
            offset = base.copy()
            offset[joint] = delta
            candidates.append((f"j{joint}_{delta:+.2f}", offset))
    rows, best_trace, best_score, best = [], None, None, None
    for name, offset in candidates:
        actions = ramp((np.arange(800)+1)*.05)[:, None] * offset / .08
        table = np.broadcast_to(actions, (args.count, 800, 17)).astype(np.float32)
        initial, trace = jax.device_get(evaluate(params, table))
        row, episodes = summarize(trace, env, seeds)
        row.update(candidate=name, offset_rad=offset.tolist(), elapsed_sec=time.monotonic()-started)
        (args.output / f"{name}_episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        rows.append(row)
        score = (row["clean_success30"], row["mean_hold"], -float(np.linalg.norm(offset)))
        if best_score is None or score > best_score:
            best_score, best, best_trace = score, row, trace
        with (args.output / "metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
    with np.load(REFERENCE, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    offset = np.asarray(best["offset_rad"])
    arrays["ctrl"] = np.clip(arrays["ctrl"] + ramp(arrays["time"])[:, None] * offset,
                             np.asarray(env._lowers), np.asarray(env._uppers))
    arrays["cylinder_calibration_delta"] = offset
    np.savez_compressed(args.output / "reference.npz", **arrays)
    np.savez_compressed(args.output / "preview_trajectories.npz", **{k: v[:4] for k,v in best_trace.items()})
    (args.output / "model.xml").write_text(env.evaluation_xml+"\n")
    report = dict(best, initial_gate=gate, candidates=len(rows), seeds=seeds,
                  source_reference=str(REFERENCE), source_reference_sha256=digest(REFERENCE),
                  reference_sha256=digest(args.output / "reference.npz"),
                  source_video="HIC20 cube tracking: CROSS-SHAPE transfer, not a cylinder video",
                  method="Single-joint simulator endpoint calibration after 1.5s, full offset at 3s",
                  first=rows[0]["clean_success30"], last=rows[-1]["clean_success30"],
                  max=max(r["clean_success30"] for r in rows), best_step=best["candidate"],
                  simulation_episodes=len(rows)*args.count,
                  total_elapsed_sec=time.monotonic()-started, status="completed")
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("BEST", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
