"""Read-only phase-wise BC fit and closed-loop target-drift diagnostics."""

import argparse
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Aero-Hand/sim_rl/mujoco_playground"))

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from train_orca_comparison import network_factory
from retarget_s09_reference import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    config = json.loads((a.run / "run_config.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(Path(config["input"]) / "model.xml"))
    half = np.diff(model.actuator_ctrlrange, axis=1).ravel() / 2
    net = network_factory(dict(state=(123,), privileged_state=(140,)), 17)
    predict = jax.jit(lambda params, obs: net.policy_network.apply(params[0], params[1], {"state": obs}))
    with np.load(config["dataset"], allow_pickle=False) as z:
        obs, target, active = z["obs"], z["action"], z["active"].astype(bool)
    split = len(obs) * 3 // 4
    stages = {"approach_0_1.5s": (0, 30), "closure_1.5_3s": (30, 60), "hold_after3s": (60, 800)}
    rows = []
    for checkpoint in sorted(a.run.glob("step_*/params.pkl")):
        params = pickle.loads(checkpoint.read_bytes())
        for name, (start, end) in stages.items():
            mask = active[split:, start:end]
            x, y = obs[split:, start:end][mask], target[split:, start:end][mask]
            logits = np.asarray(predict(params, jp.asarray(x)))
            error = (np.tanh(logits[:, :17]) - y) * half
            scale = np.asarray(jax.nn.softplus(logits[:, 17:])) + .001
            rows.append(dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), stage=name,
                frames=len(x), latent_mse=float(np.mean((logits[:, :17]-np.arctanh(np.clip(y,-.995,.995)))**2)),
                target_rms_rad=float(np.sqrt(np.mean(error**2))),
                target_abs_p95_rad=float(np.quantile(np.abs(error), .95)),
                target_max_rad=float(np.abs(error).max()), latent_std_mean=float(scale.mean())))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(rows=rows, dataset_sha256=digest(Path(config["dataset"])),
        warning="Offline validation-state error, not a causal proof or closed-loop success metric"), indent=2)+"\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
