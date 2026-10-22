"""Verify that contact-capacity reduction retains every active MJX contact."""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from prepare_hic_orca_object import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-model", type=Path, required=True)
    p.add_argument("--capped-model", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--random-samples", type=int, default=768)
    args = p.parse_args()
    if args.random_samples < 1:
        p.error("Positive stress-test sample count required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    assert jax.default_backend() == "gpu"
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_compilation_cache_dir", str(Path("artifacts/orca30/jax_cache").resolve()))
    full = mujoco.MjModel.from_xml_path(str(args.full_model))
    capped = mujoco.MjModel.from_xml_path(str(args.capped_model))
    assert np.all(full.geom_margin == 0) and np.all(capped.geom_margin == 0)
    capacity = int(capped.numeric("max_contact_points").data[0])
    with np.load(args.trace) as z:
        active = z["active"].astype(bool)
        qpos, mocap = z["qpos"][active][::32], z["mocap"][active][::32]
    n_trace = len(qpos)
    rng = np.random.default_rng(1701)
    random_q = np.tile(mujoco.MjData(full).qpos, (args.random_samples, 1))
    qids = full.jnt_qposadr[full.actuator_trnid[:, 0]]
    random_q[:, qids] = rng.uniform(*full.actuator_ctrlrange.T, size=(args.random_samples, full.nu))
    cube_id = int(full.joint("cube_freejoint").qposadr[0])
    center = full.body("cube").pos
    random_q[:, cube_id:cube_id + 3] = center + rng.uniform(-.06, .06, (args.random_samples, 3))
    random_q[:, cube_id + 3:cube_id + 7] = Rotation.random(args.random_samples, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
    random_mocap = np.tile(mocap[-1:], (args.random_samples, 1, 1))
    qpos, mocap = np.concatenate([qpos, random_q]), np.concatenate([mocap, random_mocap])
    started, results = time.monotonic(), []
    for model in (full, capped):
        device_model, data = mjx.put_model(model), mjx.make_data(model)
        def query(q, mp):
            moved = mjx.kinematics(device_model, data.replace(qpos=q, mocap_pos=mp))
            contact = mjx.collision(device_model, moved).contact
            return dict(geom=contact.geom, dist=contact.dist, margin=contact.includemargin,
                        pos=contact.pos, frame=contact.frame)
        query_batch = jax.jit(jax.vmap(query))
        chunks = []
        for i in range(0, len(qpos), 32):
            qs, ms = qpos[i:i + 32], mocap[i:i + 32]
            size = len(qs)
            if size < 32:
                qs = np.concatenate([qs, np.repeat(qs[-1:], 32 - size, axis=0)])
                ms = np.concatenate([ms, np.repeat(ms[-1:], 32 - size, axis=0)])
            chunk = jax.device_get(query_batch(jp.asarray(qs), jp.asarray(ms)))
            chunks.append({k: value[:size] for k, value in chunk.items()})
        result = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
        results.append(result)
        print("COLLISION", result["dist"].shape, flush=True)
    before, after = results
    counts, errors = [], []
    failures = []
    for i in range(len(qpos)):
        a = np.flatnonzero(before["dist"][i] < before["margin"][i])
        b = np.flatnonzero(after["dist"][i] < after["margin"][i])
        counts.append(len(a))
        if len(a) != len(b) or len(a) >= capacity:
            failures.append(dict(sample=i, full=len(a), capped=len(b), reason="count or capacity"))
            continue
        error = 0.
        for pair in np.unique(before["geom"][i, a], axis=0):
            ai = a[np.all(before["geom"][i, a] == pair, axis=1)]
            bi = b[np.all(after["geom"][i, b] == pair, axis=1)]
            if len(ai) != len(bi):
                failures.append(dict(sample=i, pair=pair.tolist(), reason="geom pair changed"))
                continue
            distances = np.linalg.norm(before["pos"][i, ai, None] - after["pos"][i, bi][None], axis=-1)
            rows, cols = linear_sum_assignment(distances)
            for key, tolerance in (("pos", 2e-6), ("dist", 2e-6), ("frame", 2e-5)):
                difference = float(np.max(np.abs(before[key][i, ai[rows]] - after[key][i, bi[cols]])))
                if key in ("pos", "dist"):
                    error = max(error, difference)
                if difference > tolerance:
                    failures.append(dict(sample=i, key=key, error=difference, reason="contact changed"))
        errors.append(error)
    counts = np.asarray(counts)
    np.savez_compressed(args.output / "samples.npz", qpos=qpos, mocap=mocap, active_counts=counts)
    report = dict(pass_active_contact_preservation=not failures, failures=failures,
                  full_model_sha256=digest(args.full_model), capped_model_sha256=digest(args.capped_model),
                  trace_sha256=digest(args.trace), capacity=capacity, full_static_slots=before["dist"].shape[1],
                  trace_samples=n_trace, random_samples=args.random_samples, max_active_contacts=int(counts.max()),
                  max_trace_contacts=int(counts[:n_trace].max()), max_random_contacts=int(counts[n_trace:].max()),
                  max_contact_position_or_distance_difference_m=max(errors, default=0.),
                  elapsed_sec=time.monotonic() - started,
                  note="Finite collision-only audit, not a global guarantee; runtime saturation and full dynamics gates remain required")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if failures:
        raise SystemExit("Active contact preservation failed")


if __name__ == "__main__":
    main()
