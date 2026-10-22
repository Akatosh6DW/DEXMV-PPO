"""Bind S09 evaluation previews to learned weights, reference, and native replay."""

import argparse
import json
from pathlib import Path
import pickle

import jax
import numpy as np

import run_s09_job as runner
import audit_s09_bc_repair
import audit_s09_replay
from orca_s09_env import OrcaS09Grasp
from retarget_s09_reference import digest


def verify_reference(env, trace):
    reset = jax.jit(env.reset)
    bank = np.asarray(env.reference_ctrl)
    error = 0.
    for i, seed in enumerate(trace["seeds"]):
        state = reset(jax.random.PRNGKey(int(seed)))
        index = int(state.info["reference_index"])
        active = trace["active"][i].astype(bool)
        np.testing.assert_array_equal(trace["reference_index"][i,active], index)
        expected = bank[index,np.flatnonzero(active)]
        actual = trace["video_base"][i,active]
        np.testing.assert_array_equal(actual, expected)
        error = max(error, float(np.max(np.abs(actual-expected))))
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    cfg = json.loads((args.run/"run_config.json").read_text())
    folder = args.run/f"step_{args.step:09d}"
    jax.config.update("jax_default_matmul_precision", "highest")
    if jax.default_backend() != cfg["backend"]:
        raise ValueError("Use the same recorded backend for reset randomness")
    binding = {}
    if cfg["method"] == "guided":
        audit_s09_bc_repair.main()
    elif cfg["method"] == "residual":
        inputs = Path(cfg["input"])
        env = OrcaS09Grasp(inputs/"model.xml", "residual", inputs/"reference.npz",
            approach_weight=cfg["approach_weight"], coverage_weight=cfg["coverage_weight"],
            orientation_mode=cfg["orientation"])
        net, _ = runner.initialize_matched(env,cfg["seed"],cfg["std"])
        params = pickle.loads((folder/"params.pkl").read_bytes())
        with np.load(folder/"preview_trajectories.npz",allow_pickle=False) as archive:
            trace = dict(archive)
        policy = runner.networks.make_inference_fn(net)(params, deterministic=True)
        infer = jax.jit(lambda obs:policy({"state":obs},jax.random.PRNGKey(0))[0])
        actual = np.asarray(infer(trace["obs"]))[trace["active"]]
        expected = trace["action"][trace["active"]]
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)
        binding = dict(max_policy_action_error=float(np.max(np.abs(actual-expected))),
            max_reference_base_error=verify_reference(env,trace),
            checkpoint_sha256=digest(folder/"params.pkl"),
            reference_sha256=digest(inputs/"reference.npz"),
            normalized_action_tolerance=1e-5,
            requested_residual_rms_rad=float(np.sqrt(np.mean(trace["requested_residual"][trace["active"]]**2))))
        audit_s09_replay.main()
    elif cfg["method"] == "reference":
        audit_s09_replay.main()
        binding = dict(policy_checkpoint=False, method="reference only")
    else:
        parser.error("This audit is scoped to repaired guided, residual, and reference runs")
    native = json.loads((folder/"native_replay_audit.json").read_text())
    if native["status"] != "verified":
        raise ValueError("Native replay was not verified")
    if cfg["method"] == "guided":
        binding = {key:native[key] for key in ("checkpoint_sha256", "max_policy_action_error", "action_check")}
    result = dict(status="verified", method=cfg["method"], policy_binding=binding,
        native_audit_sha256=digest(folder/"native_replay_audit.json"),
        trace_sha256=digest(folder/"preview_trajectories.npz"),
        scope="All active steps in saved previews only; not all unsaved evaluation episodes")
    (folder/"improvement_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
