"""Check all prespecified test reset geometries without rolling out a policy."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from orca_suite_env import BENCHMARKS, OrcaSuiteGrasp
from train_orca_acquisition import MODEL, REFERENCE
from probe_hic53_orca import cube_penetration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    seeds = protocol["test_seeds"]
    results = []
    for name, settings in BENCHMARKS.items():
        env = OrcaSuiteGrasp(MODEL, "pure", REFERENCE, benchmark=name)
        states = jax.device_get(jax.jit(jax.vmap(env.reset))(
            jax.vmap(jax.random.PRNGKey)(jp.array(seeds))))
        data = mujoco.MjData(env.mj_model)
        depths = []
        for q, m in zip(states.data.qpos, states.data.mocap_pos):
            data.qpos[:], data.mocap_pos[:] = q, m
            mujoco.mj_forward(env.mj_model, data)
            depths.append(cube_penetration(env.mj_model, data))
        assert states.obs["state"].shape == (200, 123)
        assert np.isfinite(states.obs["state"]).all()
        assert (states.obs["state"][:, 94] == 0).all()
        results.append(dict(benchmark=name, settings=settings, seeds=seeds,
                            reset_depths_m=depths, max_reset_depth_m=max(depths),
                            cube_mass_kg=float(env.mj_model.body("cube").mass[0]),
                            cube_friction=env.mj_model.geom("cube").friction.tolist(),
                            source_qpos_sha256=hashlib.sha256(states.data.qpos.tobytes()).hexdigest(),
                            passed=max(depths) <= .003))
        print(name, results[-1]["max_reset_depth_m"], flush=True)
    report = dict(passed=all(row["passed"] for row in results), policy_rollouts=0,
                  cases=results, protocol_sha256=hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
                  note="All final test reset geometries only; no success outcomes inspected or cases excluded.")
    (root / "reset_geometry_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    assert report["passed"], "Geometry gate failed; do not silently filter final test cases"


if __name__ == "__main__":
    main()
