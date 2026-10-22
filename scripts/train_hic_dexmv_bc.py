#!/usr/bin/env python3
"""Train a phase-conditioned motion prior using DexMV/mjrl behavior cloning.

This is an open-loop reference policy, NOT a demonstrated physical grasp success.
PPO will supply the state-feedback residual in the target-hand simulator.
"""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mjrl.algos.behavior_cloning import BC
from mjrl.policies.gaussian_mlp import MLP


def features(time, duration):
    phase = np.asarray(time) / duration
    return np.stack([phase, np.sin(np.pi * phase), np.cos(np.pi * phase)], axis=-1).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("artifacts/cube20/hic20/dexmv_retarget/adroit_cube.npz"))
    p.add_argument("--output", type=Path, default=Path("artifacts/cube20/hic20/dexmv_bc"))
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--export-dt", type=float, default=0.05)
    a = p.parse_args()
    torch.set_num_threads(2)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    d = np.load(a.input)
    q = d["adroit_qpos"].astype(np.float32)
    time, limits = d["time"], d["joint_limits"]
    mid, span = limits.mean(1), np.diff(limits, axis=1).ravel() / 2
    obs = features(time, time[-1])
    acts = ((q - mid) / span).astype(np.float32)
    # Interior held-out frames measure interpolation only, not novel-demo generalization.
    valid = np.arange(len(time)) % 5 == 2
    paths = [{"observations": obs[~valid], "actions": acts[~valid]}]
    policy = MLP(SimpleNamespace(observation_dim=3, action_dim=30), hidden_sizes=(128, 128), seed=a.seed)
    bc = BC(paths, policy, epochs=1, batch_size=32, lr=1e-3)
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    best = float("inf")
    for epoch in range(a.epochs + 1):
        with torch.no_grad():
            predicted = policy.model(torch.from_numpy(obs)).numpy()
        mse = float(np.mean((predicted[valid] - acts[valid]) ** 2))
        predicted_q = np.clip(predicted, -1, 1) * span + mid
        rows.append({"epoch": epoch, "train_mse": float(np.mean((predicted[~valid] - acts[~valid]) ** 2)),
                     "validation_mse": mse,
                     "validation_finger_rmse_rad": float(np.sqrt(np.mean((predicted_q[valid, 8:] - q[valid, 8:]) ** 2)))})
        if mse < best:
            best, best_epoch = mse, epoch
            best_policy = pickle.dumps(policy)
        if epoch % 100 == 0:
            print(rows[-1], flush=True)
        if epoch == a.epochs:
            break
        # Use the existing BC loss and optimizer; log held-out error each epoch.
        for _ in range(max(1, len(obs[~valid]) // bc.mb_size)):
            indices = np.random.choice(np.flatnonzero(~valid), bc.mb_size)
            bc.optimizer.zero_grad()
            loss = bc.loss(obs[indices], acts[indices])
            loss.backward()
            bc.optimizer.step()
    policy = pickle.loads(best_policy)
    policy.set_param_values(policy.get_param_values(), set_new=True, set_old=True)
    with (a.output / "policy.pickle").open("wb") as f:
        pickle.dump(policy, f)
    with (a.output / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    export_time = np.arange(int(np.floor(time[-1] / a.export_dt)) + 1) * a.export_dt
    with torch.no_grad():
        exported = policy.model(torch.from_numpy(features(export_time, time[-1]))).numpy()
    exported_q = np.clip(exported, -1, 1) * span + mid
    meta = {"algorithm": "DexMV/mjrl BC (supervised phase-to-motion prior)",
            "source": str(a.input.resolve()), "source_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
            "seed": a.seed, "best_epoch": best_epoch,
            "first": rows[0], "last": rows[-1], "best": rows[best_epoch],
            "heldout_scope": "interleaved frames of the same clip, NOT unseen demonstrations",
            "physical_grasp_verified": False, "export_dt": a.export_dt}
    np.savez_compressed(a.output / "learned_prior.npz", adroit_qpos=exported_q, time=export_time,
                        dt=np.array(a.export_dt), metadata_json=np.array(json.dumps(meta)))
    (a.output / "report.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
