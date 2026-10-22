#!/usr/bin/env python3
"""Same Orca hand and object mass/friction, three supported collision shapes."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco

from diagnose_orca_grasp import SCENE

SHAPES = {"cube": ("box", ".018 .018 .018", .018),
          "sphere": ("sphere", ".021", .021),
          "cuboid": ("box", ".021 .018 .018", .018)}


def build(output, shape):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    with tempfile.TemporaryDirectory() as tmp:
        flat = Path(tmp)/"flat.xml"
        mujoco.mj_saveLastXML(str(flat), model)
        tree = ET.parse(flat)
    root = tree.getroot()
    root.set("model", f"Orca shared grasp {shape}")
    geom = root.find(".//geom[@name='cube']")
    kind, size, half_height = SHAPES[shape]
    geom.set("type", kind)
    geom.set("size", size)
    geom.set("mass", ".035")
    ET.indent(tree)
    tree.write(output, encoding="unicode")
    mujoco.MjModel.from_xml_path(str(output))
    return dict(shape=shape, geom_type=kind, size=size, half_height=half_height,
                mass_kg=.035, sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {shape: build(args.output/f"{shape}.xml", shape) for shape in SHAPES}
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
