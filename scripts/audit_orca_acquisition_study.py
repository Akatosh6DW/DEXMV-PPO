"""Audit experiment completeness, paired BC initialization, and actual test resets."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from orca_acquisition_env import OrcaAcquisitionGrasp
from orca_guidance_recheck import score
from probe_hic53_orca import cube_penetration
from train_orca_acquisition import MODEL, REFERENCE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    jax.config.update("jax_pmap_shmap_merge", False)
    for path, expected in protocol["source_hashes"].items():
        assert digest(path) == expected, path
    checks = []
    for job in protocol["jobs"]:
        directory = root / job["name"]
        summary = json.loads((directory / "summary.json").read_text())
        meta = json.loads((directory / "run_config.json").read_text())
        assert summary["status"] == "completed"
        assert meta["model_sha256"] == digest(MODEL)
        assert meta["reference_sha256"] == digest(REFERENCE)
        for key in ("method", "seed", "steps", "std", "learning_rate", "jitter", "eval_start", "eval_count"):
            if key in job["args"]:
                assert meta[key] == job["args"][key], (job["name"], key)
        if "checkpoint" in job["args"]:
            assert meta["checkpoint_sha256"] == digest(job["args"]["checkpoint"])
        checks.append(dict(job=job["name"], completed=True, final_step=summary["final_step"]))
    paired_bc = []
    for seed in protocol["independent_training_seeds"]:
        a = pickle.loads((root / f"train/guided_s{seed}/bc_params.pkl").read_bytes())
        b = pickle.loads((root / f"train/bc_s{seed}/params.pkl").read_bytes())
        for left, right in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
            np.testing.assert_array_equal(left, right)
        paired_bc.append(dict(seed=seed, identical=True))
    resets = []
    for jitter in (.002, .004):
        env = OrcaAcquisitionGrasp(MODEL, "pure", REFERENCE, jitter=jitter)
        states = jax.device_get(jax.jit(jax.vmap(env.reset))(
            jax.vmap(jax.random.PRNGKey)(jp.array(protocol["test_seeds"]))))
        data = mujoco.MjData(env.mj_model)
        values = []
        for q, m in zip(states.data.qpos, states.data.mocap_pos):
            data.qpos[:] = q; data.mocap_pos[:] = m
            mujoco.mj_forward(env.mj_model, data)
            values.append(cube_penetration(env.mj_model, data))
        assert max(values) <= .003
        resets.append(dict(jitter=jitter, seeds=protocol["test_seeds"], max_penetration_m=max(values),
                           invalid_initial_count=sum(x>.003 for x in values)))
    previews = []
    environments = {}
    for directory in sorted((root / "eval").iterdir()):
        if not directory.is_dir():
            continue
        meta = json.loads((directory / "run_config.json").read_text())
        arm = "residual" if meta["method"]=="residual" else "pure"
        key = (arm, meta["jitter"])
        if key not in environments:
            environments[key] = OrcaAcquisitionGrasp(MODEL, arm, REFERENCE, jitter=meta["jitter"])
        trace_path = directory / "step_000000000/preview_trajectories.npz"
        with np.load(trace_path) as z:
            trace = dict(z)
        _, holds, successes = score(trace, environments[key])
        episodes = json.loads((directory / "step_000000000/episodes.json").read_text())
        np.testing.assert_allclose(holds, [r["longest_hold_sec"] for r in episodes[:len(holds)]])
        np.testing.assert_array_equal(successes, [r["success30"] for r in episodes[:len(holds)]])
        previews.append(dict(run=directory.name, episodes_checked=len(holds), trace_sha256=digest(trace_path)))
    report = dict(passed=True, jobs=checks, paired_bc=paired_bc, actual_test_resets=resets,
                  saved_trace_checks=previews, protocol_sha256=digest(root / "protocol.json"),
                  scope="All jobs and paired initializations; all actual test resets; saved first four trajectories per evaluation. Other episodes retained as metrics and reproducible checkpoints/seeds, not full traces.")
    (root / "audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
