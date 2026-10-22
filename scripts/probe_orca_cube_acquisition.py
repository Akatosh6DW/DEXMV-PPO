"""Native development probe: open-reset acquisition and bounded video refinement."""

import argparse
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from diagnose_orca_grasp import model_indices
from orca_residual_controls import load_reference
from prepare_hic_orca_object import digest
from probe_hic53_orca import native_rollout, cube_penetration


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
REFERENCE = MODEL.parent / "reference.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opening", type=float, default=.35)
    parser.add_argument("--opening-mode", choices=("all", "flex"), default="flex")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--bound", type=float, default=.12)
    parser.add_argument("--seed", type=int, default=701)
    args = parser.parse_args()
    if not 0 <= args.opening <= 1 or not 0 < args.bound <= .15:
        parser.error("Invalid opening or bounded refinement")
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    with np.load(REFERENCE) as z:
        packet = {k: z[k] for k in z.files}
    low, high = model.actuator_ctrlrange.T
    neutral = np.clip(np.r_[-.25, np.zeros(16)], low, high)
    initial = packet["initial_ctrl"].copy()
    opened = np.arange(17) if args.opening_mode == "all" else np.array([11, 12, 14, 16])
    initial[opened] = (1 - args.opening) * initial[opened] + args.opening * neutral[opened]
    initial = np.clip(initial, low, high)
    reference, _ = load_reference(REFERENCE, low, high, initial, .05, 800, .1)
    qids, cube_id, _ = model_indices(model)
    data = mujoco.MjData(model)
    data.qpos[qids] = initial
    data.qpos[cube_id:cube_id+7] = np.r_[packet["object_position"], packet["object_quaternion"]]
    data.mocap_pos[0] = packet["object_position"] - [0, 0, .053/2 + .004]
    mujoco.mj_forward(model, data)
    if cube_penetration(model, data) > .003:
        raise ValueError("Invalid initial hand penetration")
    rng = np.random.default_rng(args.seed)
    selected = np.arange(10, 17)
    mean, std = np.zeros(7), np.full(7, args.bound / 2)
    records, best = [], None
    started = time.monotonic()
    for generation in range(args.generations):
        candidates = np.clip(rng.normal(mean, std, (args.population, 7)), -args.bound, args.bound)
        candidates[0] = mean
        if generation == 0:
            candidates[0] = 0
        ranked = []
        for i, delta in enumerate(candidates):
            bias = np.zeros(17)
            bias[selected] = delta
            controls = np.clip(reference + np.minimum(np.arange(1, 801) * .05 / 1.5, 1)[:, None] * bias, low, high)
            _, result = native_rollout(model, packet["object_position"], packet["object_quaternion"],
                                       controls, args.seed, initial_ctrl=initial)
            quality = (result["reset_penetration_m"] <= .003 and result["max_penetration_m"] <= .003)
            value = result["best_hold_sec"] + 10 * result["success30"] - 2000 * max(result["max_penetration_m"]-.003, 0)
            value -= .02 * np.linalg.norm(delta) + .0001 * result["max_force_n"]
            row = dict(generation=generation, candidate=i, delta=delta.tolist(), value=value,
                       geometry_quality=quality, elapsed_sec=time.monotonic()-started, **result)
            records.append(row)
            ranked.append(row)
            if best is None or value > best["value"]:
                best = row
                np.savez_compressed(args.output / "candidate_reference.npz", **dict(packet,
                                    ctrl=np.vstack([initial, controls]), time=np.arange(801)*.05, initial_ctrl=initial,
                                    refinement_delta=bias, initial_opening=np.array(args.opening)))
            print(json.dumps(row), flush=True)
        elite = sorted(ranked, key=lambda r: r["value"], reverse=True)[:max(4, args.population//4)]
        mean = np.mean([r["delta"] for r in elite], axis=0)
        std = np.maximum(np.std([r["delta"] for r in elite], axis=0), .01)
        (args.output / "search.json").write_text(json.dumps(records, indent=2)+"\n")
    with np.load(args.output / "candidate_reference.npz") as z:
        controls = z["ctrl"][1:]
    validation = []
    for seed in range(710, 718):
        trace, report = native_rollout(model, packet["object_position"], packet["object_quaternion"],
                                       controls, seed, initial_ctrl=initial)
        validation.append(report)
        if seed == 710:
            np.savez_compressed(args.output / "validation_seed710.npz", **trace)
    report = dict(stage="development_video_refinement_not_PPO", model=str(MODEL), reference=str(REFERENCE),
                  model_sha256=digest(MODEL), original_reference_sha256=digest(REFERENCE),
                  source_video_count=1, source_sequence=20, source_frames=[30, 95],
                  opening=args.opening, opening_mode=args.opening_mode, bound=args.bound, initial_ctrl=initial.tolist(), best=best,
                  validation=validation, validation_success=float(np.mean([r["success30"] for r in validation])),
                  correction="simulator-selected bounded joint bias, not a newly observed video movement",
                  source_sha256=digest(Path(__file__)))
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    shutil.copyfile(__file__, args.output / "probe_source.py")
    print("VALIDATED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
