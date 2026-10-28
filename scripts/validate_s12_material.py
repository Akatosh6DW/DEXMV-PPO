"""Independent compression/unload test of the same tetrahedral material."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from s12_soft import SoftConfig, add_flex, numbers, tetra_cylinder, tetra_metrics


def compression(cfg):
    root = ET.Element("mujoco", model="S12 compression material validation")
    ET.SubElement(root, "option", timestep=str(cfg.timestep), gravity="0 0 0", integrator="implicit",
                  jacobian="sparse", cone="elliptic", iterations="50")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", limited="false", damping="0", frictionloss="0", armature="0")
    ET.SubElement(default, "geom", contype="4", conaffinity="2", friction="0 0 0",
                  solref=".004 1", solimp=".95 .99 .001")
    world = ET.SubElement(root, "worldbody")
    points, tet = tetra_cylinder(cfg)
    body = ET.SubElement(world, "body", name="material_frame")
    add_flex(body, cfg, points, tet)
    bottom = -cfg.half_length-cfg.contact_radius
    ET.SubElement(world, "geom", name="floor", type="plane", size=".1 .1 .01", pos=numbers([0, 0, bottom]))
    top_z = cfg.half_length+cfg.contact_radius+.003
    top = ET.SubElement(world, "body", name="plate", mocap="true", pos=numbers([0, 0, top_z]))
    ET.SubElement(top, "geom", name="plate_geom", type="box", size=".06 .06 .003")
    xml = ET.tostring(root, encoding="unicode")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    tet = m.flex_elem.reshape(-1, 4)
    rest = d.flexvert_xpos.copy()
    inverse = np.linalg.inv((rest[tet[:, 1:]]-rest[tet[:, :1]]).transpose(0, 2, 1))
    rows = []
    for step in range(round(1.5/cfg.timestep)):
        t = step*cfg.timestep
        fraction = min(t/.4, 1.) if t<.7 else max(1-(t-.7)/.3, 0.)
        d.mocap_pos[0, 2] = top_z - .05*(2*cfg.half_length)*fraction
        mujoco.mj_step(m, d)
        if np.any(d.warning.number) or not np.isfinite(d.qpos).all():
            raise FloatingPointError(f"Material unstable at t={t}")
        if step % max(1, round(.005/cfg.timestep)) == 0:
            load = 0.
            for i, c in enumerate(d.contact[:d.ncon]):
                if m.geom("plate_geom").id in c.geom:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, force)
                    load += force[0]
            strain, jac, volume = tetra_metrics(d.flexvert_xpos, tet, inverse)
            rows.append([d.time, load, strain, jac, volume, np.ptp(d.flexvert_xpos[:, 2])])
    rows = np.array(rows)
    held = (rows[:, 0]>.55)&(rows[:, 0]<.69)
    result = dict(config=asdict(cfg), compression_force_n=float(np.mean(rows[held, 1])),
                  loaded_strain=float(np.mean(rows[held, 2])), recovery_strain=float(rows[-1, 2]),
                  min_volume_ratio=float(rows[:, 3].min()), peak_strain=float(rows[:, 2].max()),
                  final_height_m=float(rows[-1, 5]), rest_height_m=2*cfg.half_length)
    return result, rows, xml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timestep", type=float, default=.0002)
    p.add_argument("--damping", type=float, default=.002)
    p.add_argument("--young", type=float, nargs="+", default=[5000., 10000., 20000.])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    for young in a.young:
        cfg = SoftConfig(young=young, timestep=a.timestep, damping=a.damping)
        try:
            result, trace, xml = compression(cfg)
        except (FloatingPointError, ValueError) as error:
            results.append(dict(config=asdict(cfg), error=str(error)))
            (a.output/"report.json").write_text(json.dumps(results, indent=2)+"\n")
            raise
        results.append(result)
        (a.output/"report.json").write_text(json.dumps(results, indent=2)+"\n")
        np.savetxt(a.output/f"E{int(young)}_compression.csv", trace, delimiter=",",
                   header="time,normal_force_n,max_principal_strain,min_volume_ratio,mean_volume_ratio,height_m", comments="")
        (a.output/f"E{int(young)}.xml").write_text(xml)
        print(json.dumps(result), flush=True)
    (a.output/"report.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
