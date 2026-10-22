"""Recompute saved hold criteria and deterministically replay position-only tests."""

import json

import mujoco
import numpy as np

from probe_s08_position import BASE, OUT, SOURCE, reset_contacts, save_json
from probe_s08_wrap import model_indices, rollout
from prepare_cylinder_s02_source import digest


def main():
    reports = json.loads((OUT / "report.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert digest(OUT / "reference.npz") == selection["reference_sha256"]
    assert digest(SOURCE) == protocol["source_sha256"]
    assert digest(BASE) == protocol["base_model_sha256"]
    m = mujoco.MjModel.from_xml_path(str(BASE))
    assert m.neq == 0
    _, oid, _ = model_indices(m)
    source, packet = dict(np.load(SOURCE)), dict(np.load(OUT / "reference.npz"))
    for key in source:
        if key != "object_position":
            np.testing.assert_array_equal(source[key], packet[key])
    xml_model = mujoco.MjModel.from_xml_path(str(OUT / "model.xml"))
    np.testing.assert_allclose(xml_model.qpos0[oid:oid+7], np.r_[packet["object_position"], packet["object_quaternion"]])
    count = 0
    for label, p in [("baseline", source), ("position_only", packet)]:
        for row in reports[label]["tests"]:
            with np.load(OUT / label / f"trace_{row['seed']}.npz") as z:
                trace = dict(z)
            rng = np.random.default_rng(row["seed"])
            spawn = p["object_position"]+rng.uniform(-.002,.002,3)*[1,1,0]
            times = np.arange(1, len(trace["hold"])+1)*.05
            loads = trace["loads"]
            wrap = (loads[:,0]>.02)&(loads[:,1]>.02)&((loads[:,2:]>.02).sum(axis=1)>=2)
            finite = np.isfinite(trace["qpos"]).all(axis=1)&np.isfinite(trace["qvel"]).all(axis=1)
            alive = finite & (trace["qpos"][:,oid+2] >= spawn[2]-.055)
            valid = alive & (times >= 3) & (trace["qpos"][:,oid+2] > spawn[2]-.015) & wrap & ~trace["forbidden"] & (trace["penetration"]<=.003)
            running, hold = 0, []
            for v in valid:
                running = running+1 if v else 0
                hold.append(running*.05)
            np.testing.assert_array_equal(trace["hold"], hold)
            np.testing.assert_array_equal(trace["force_wrap"], wrap)
            np.testing.assert_array_equal(trace["alive"], alive)
            reset = reset_contacts(m, p, row["seed"])
            assert reset["depth_m"] == row["reset_penetration_m"]
            released = times >= 3
            success = bool(max(hold)>=30 and valid[-1] and reset["depth_m"]<=.001
                and not trace["forbidden"][released].any() and (trace["penetration"][released]<=.003).all())
            assert success == row["success30"]
            assert row["duration_sec"] == len(hold)*.05
            if label == "position_only":
                replay_report, replay_trace = rollout(m,p,row["seed"],40)
                assert replay_report == row
                for key in trace:
                    np.testing.assert_array_equal(trace[key], replay_trace[key])
            count += 1
            print("AUDIT",label,row["seed"],success,flush=True)
    save_json(OUT / "validation_audit.json", dict(episodes_checked=count, identical_replay=True,
        replayed_episodes=len(reports["position_only"]["tests"]), trace_criteria_checked=True,
        controls_unchanged=True, model_defaults_match=True, source_hashes_match=True,
        reference_sha256=digest(OUT / "reference.npz")))


if __name__ == "__main__":
    main()
