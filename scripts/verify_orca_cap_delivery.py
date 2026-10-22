#!/usr/bin/env python3
"""Verify saved cap scores against full states and trace/video provenance."""

import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from orca_cap_metrics import hold_seconds

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/orca_cap"


def validate_trace(path, model):
    with np.load(path) as z:
        trace = {k: z[k] for k in z.files}
    turn = model.joint("cap_turn").qposadr[0]
    lift = model.joint("cap_lift").qposadr[0]
    np.testing.assert_allclose(trace["qpos"][..., turn], trace["angle"], atol=1e-7)
    np.testing.assert_allclose(trace["qpos"][..., lift], trace["lift"], atol=1e-7)
    np.testing.assert_allclose(np.abs(trace["lift"]-trace["angle"]*.002/(2*np.pi)), trace["thread_error"], atol=1e-7)
    return hold_seconds(trace)


def main():
    checked = []
    model = mujoco.MjModel.from_xml_path(str(BASE/"c09_ppo_reset_fixed/model.xml"))
    assert model.nu == 17
    for run in ("c09_ppo_reset_fixed", "c12_video_bc_ppo"):
        with (BASE/run/"audited_metrics.csv").open() as f:
            for row in csv.DictReader(f):
                path = BASE/run/f"step_{int(row['step']):09d}"/"trajectories.npz"
                holds = validate_trace(path, model)
                np.testing.assert_allclose(np.mean(holds >= 1), float(row["success"]))
                np.testing.assert_allclose(holds.mean(), float(row["mean_hold"]))
                checked.append(str(path.relative_to(ROOT)))
    for directory in ("c16_paired_heldout", "c18_repeat_paired_heldout"):
        reports = json.loads((BASE/directory/"summary.json").read_text())
        assert len(reports) == 4
        for name, report in reports.items():
            policy = Path(report["checkpoint"])
            assert hashlib.sha256(policy.read_bytes()).hexdigest() == report["checkpoint_sha256"]
            path = BASE/directory/name/"trajectories.npz"
            holds = validate_trace(path, model)
            np.testing.assert_allclose(holds, report["hold_sec"])
            np.testing.assert_allclose(np.mean(holds >= 1), report["summary"]["success"])
            checked.append(str(path.relative_to(ROOT)))
    for directory in ("c13_scratch_native_heldout", "c17_guided_native_heldout"):
        path = BASE/directory/"trajectories.npz"
        holds = validate_trace(path, model)
        report = json.loads((BASE/directory/"report.json").read_text())
        np.testing.assert_allclose(np.mean(holds >= 1), report["success_rate"])
        np.testing.assert_allclose(holds, [r["best_hold_sec"] for r in report["records"]])
        checked.append(str(path.relative_to(ROOT)))
    for directory in ("scratch_mjx_videos", "guided_mjx_videos", "scratch_native_videos"):
        report = json.loads((BASE/directory/"report.json").read_text())
        assert report["physics_arrays_match"] and report["same_saved_states_both_modes"]
        assert report["frames"] == 240 and report["fps"] == 20
        assert hashlib.sha256(Path(report["trace"]).read_bytes()).hexdigest() == report["trace_sha256"]
        for mode in ("collision", "stl"):
            path = BASE/directory/f"{mode}.mp4"
            assert path.stat().st_size > 10000
            checked.append(str(path.relative_to(ROOT)))
    manifest = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in checked}
    for name in ("docs/orca_experiment_explained.md", "docs/orca_cap_experiment.md"):
        manifest[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    report = dict(status="verified_saved_states_scores_and_video_provenance", files=manifest,
                  notes="Video decoding checked separately; known GPU repeatability limitations remain documented.")
    (BASE/"delivery_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(status=report["status"], verified_files=len(manifest))))


if __name__ == "__main__":
    main()
