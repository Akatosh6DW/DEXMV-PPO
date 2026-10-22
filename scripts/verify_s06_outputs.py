"""Recompute all held-out outcomes and verify the selected weights and videos."""

import json
from pathlib import Path
import re

import numpy as np

from prepare_cylinder_s02_source import digest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts/orca_benchmark_suite/S06/E01_independent"
DELIVERY = ROOT / "deliverables/S06_圆柱三指抓持_2026-10-12"


def main():
    summary = json.loads((EVAL / "summary.json").read_text())
    selection = json.loads((EVAL / "selection.json").read_text())
    assert digest(EVAL / "selection.json") == summary["selection_sha256"]
    for name, item in selection["selected"].items():
        assert digest(item["checkpoint"]) == item["sha256"]
        if name != "bc_only":
            assert item["development"]["step"] > 0
    initial, counts = None, {}
    for method in ("reference", "residual", "guided", "bc_only"):
        path = EVAL / method / "step_000000000"
        config = json.loads((path.parent / "run_config.json").read_text())
        assert config["backend"] == "gpu" and config["physics_backend"] == "native_mujoco_callback"
        for field in ("model", "reference"):
            assert digest(config[field]) == selection[field+"_sha256"]
        with np.load(path / "initial_states.npz") as z:
            np.testing.assert_array_equal(z["seeds"], selection["seeds"])
            if initial is None:
                initial = z["qpos"].copy()
            else:
                np.testing.assert_array_equal(initial, z["qpos"])
        rows = json.loads((path / "episodes.json").read_text())
        with np.load(path / "preview_trajectories.npz") as z:
            assert z["qpos"].shape == (16, 800, 24)
            assert all(np.isfinite(z[k]).all() for k in z.files)
            pair = (z["pad_audit"][..., :2] > 0).all(-1) & ~(z["pad_audit"][..., 2:4] > 0).any(-1)
            np.testing.assert_array_equal(pair, z["pad_pair"].astype(bool))
            stable = (pair & (z["substep_forbidden"] == 0) & (z["substep_penetration"] <= .003)
                      & (z["broad_hold"] > 0) & z["alive"].astype(bool) & z["active"].astype(bool))
            hold = np.zeros((16, 800))
            for t in range(800):
                hold[:, t] = np.where(stable[:, t], (hold[:, t-1] if t else 0)+.05, 0)
            np.testing.assert_allclose(hold, z["hold"], atol=5e-6)
            success = []
            for i, row in enumerate(rows):
                released = z["active"][i].astype(bool) & (z["released"][i] > 0)
                clean = not np.any(z["substep_forbidden"][i, released] > 0)
                depth = float(np.max(z["substep_penetration"][i, released], initial=0))
                passed = bool(hold[i].max() >= 30 and hold[i, -1] > 0 and clean and depth <= .003)
                assert passed == row["success30"]
                success.append(passed)
            counts[method] = sum(success)
            assert counts[method] == summary["methods"][method]["successes"]
    media = json.loads((DELIVERY / "来源与校验.json").read_text())
    assert media["selection_sha256"] == summary["selection_sha256"]
    for video in media["videos"]:
        assert digest(DELIVERY / video["file"]) == video["sha256"]
        assert video["decoded_without_errors"]
        assert video["probe"]["streams"][0]["width"] == 960
        assert video["probe"]["streams"][0]["height"] == 720
    assert len(media["videos"]) == 6
    docs = list((ROOT / "docs/experiments/S06").glob("*.md")) + [DELIVERY / "目录.md"]
    for doc in docs:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
            if not target.startswith(("https://", "http://", "#")):
                assert (doc.parent / target).exists(), (doc, target)
    result = dict(status="verified", recomputed_episodes=64, paired_initial_states=True,
                  success_counts=counts, weights_and_inputs_hash_verified=True,
                  videos=6, video_hashes_and_decode_records_verified=True,
                  document_links_checked=len(docs), selection_sha256=summary["selection_sha256"],
                  limitation="Verifies these saved runs and exports; not hardware validity or broad statistical superiority")
    (EVAL / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
