"""Replay frozen wrap references with physics-warning and trace checks."""

import csv
import json
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from probe_s08_wrap import BASE, ROOT, rollout
from orca_cylinder_s02_env import cylinder_half_height
from prepare_cylinder_s02_source import digest

OUT = ROOT / "artifacts/orca_benchmark_suite/S08_wrap_refined"


def main():
    results, flat = [], []
    original_step = mujoco.mj_step
    for direction in ("longitudinal", "transverse"):
        folder = OUT / direction
        with np.load(folder / "reference.npz") as z:
            packet = dict(z)
        # Generated XML defaults must agree with the explicit NPZ reset pose.
        tree = ET.parse(BASE)
        body = tree.find(".//body[@name='cube']")
        body.set("pos", " ".join(map(str, packet["object_position"])))
        body.set("quat", " ".join(map(str, packet["object_quaternion"])))
        support = packet["object_position"]-[0, 0, cylinder_half_height(packet["object_quaternion"])+.004]
        tree.find(".//body[@name='cube_support']").set("pos", " ".join(map(str, support)))
        tree.write(folder / "model.xml", encoding="unicode")
        m = mujoco.MjModel.from_xml_path(str(BASE))
        source = json.loads((folder / "report.json").read_text())
        assert digest(folder / "reference.npz") == json.loads((folder / "selection.json").read_text())["reference_sha256"]
        for row in source["tests"]:
            warnings = np.zeros(len(mujoco.MjData(m).warning), dtype=int)
            def audited_step(model, data):
                original_step(model, data)
                warnings[:] = np.maximum(warnings, data.warning.number)
                assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            with patch("probe_s08_wrap.mujoco.mj_step", side_effect=audited_step):
                report, trace = rollout(m, packet, row["seed"], 40.)
            assert report == row
            with np.load(folder / f"trace_{row['seed']}.npz") as z:
                for key in trace:
                    np.testing.assert_array_equal(trace[key], z[key])
            assert not warnings.any(), warnings
            flat.append(dict(direction=direction, **{k:v for k,v in row.items() if k!="final_loads"},
                             **{f"load_{i}_n":v for i,v in enumerate(row["final_loads"])}, warnings=int(warnings.sum())))
        results.append(dict(direction=direction, episodes=len(source["tests"]), successes=source["successes"],
            identical_replay=True, warning_free=True, model_sha256=digest(folder / "model.xml"),
            reference_sha256=digest(folder / "reference.npz"), negative_control=source["negative_control"]))
    with (OUT / "episodes.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (OUT / "validation_audit.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
