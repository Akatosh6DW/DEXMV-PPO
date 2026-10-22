"""Create a versioned solver variant without changing geometry or demonstration."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from prepare_hic_orca_object import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=32)
    p.add_argument("--cone", choices=("elliptic", "pyramidal"), default="elliptic")
    p.add_argument("--max-contact-points", type=int)
    p.add_argument("--solver", choices=("Newton", "CG"))
    args = p.parse_args()
    if args.iterations < 1:
        p.error("Iteration count must be positive")
    if args.max_contact_points is not None and args.max_contact_points < 16:
        p.error("Contact capacity requires at least 16 slots and a separate saturation audit")
    if (args.model.parent / "mount_transform.json").exists():
        p.error("Use an unmodified mounting baseline; transformed variants require a separate proof")
    tree = ET.parse(args.model)
    option = tree.find("option")
    option.set("iterations", str(args.iterations))
    option.set("cone", args.cone)
    if args.solver is not None:
        option.set("solver", args.solver)
    if args.max_contact_points is not None:
        custom = tree.find("custom")
        if custom is None:
            custom = ET.SubElement(tree.getroot(), "custom")
        numeric = custom.find("numeric[@name='max_contact_points']")
        if numeric is None:
            numeric = ET.SubElement(custom, "numeric", name="max_contact_points")
        numeric.set("data", str(args.max_contact_points))
    for mesh in tree.findall("asset/mesh"):
        path = Path(mesh.get("file"))
        if not path.is_absolute():
            mesh.set("file", str((args.model.parent / path).resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    output_model, output_reference = args.output / "cube.xml", args.output / "reference.npz"
    tree.write(output_model, encoding="unicode")
    output_reference.write_bytes(args.reference.read_bytes())
    old, new = [mujoco.MjModel.from_xml_path(str(path)) for path in (args.model, output_model)]
    count, metadata_changes = 0, []
    numeric_metadata = {"_sizes", "numeric_adr", "numeric_data", "numeric_size", "name_numericadr", "names", "names_map"}
    for name in dir(old):
        before, after = getattr(old, name), getattr(new, name)
        if isinstance(before, np.ndarray):
            if args.max_contact_points is not None and name in numeric_metadata:
                if not np.array_equal(before, after):
                    metadata_changes.append(name)
                continue
            np.testing.assert_array_equal(before, after, err_msg=name)
            count += 1
    for name in dir(old):
        before, after = getattr(old, name), getattr(new, name)
        if name == "_address":
            continue
        if isinstance(before, (int, float)) and before != after:
            if args.max_contact_points is None or name not in ("nbuffer", "nnames", "nnames_map", "nnumeric", "nnumericdata"):
                raise ValueError("Unexpected model dimension change: " + name)
            metadata_changes.append(name)
    for name in dir(old.opt):
        if name.startswith("_") or name in ("cone", "iterations") or (name == "solver" and args.solver is not None):
            continue
        before, after = getattr(old.opt, name), getattr(new.opt, name)
        if isinstance(before, (int, float, np.ndarray)):
            np.testing.assert_array_equal(before, after, err_msg="opt." + name)
    proxy = json.loads((args.model.parent / "object_proxy_report.json").read_text())
    proxy.update(parent_model_sha256=digest(args.model), model_sha256=digest(output_model))
    (args.output / "object_proxy_report.json").write_text(json.dumps(proxy, indent=2) + "\n")
    report = dict(kind="solver-only variant", trainable=False, parent_model=str(args.model.resolve()),
                  parent_model_sha256=digest(args.model), parent_reference_sha256=digest(args.reference),
                  model_sha256=digest(output_model), reference_sha256=digest(output_reference),
                  model_arrays_unchanged=count, iterations=args.iterations, cone=args.cone,
                  max_contact_points=args.max_contact_points, numeric_metadata_changes=metadata_changes,
                  solver=int(new.opt.solver),
                  impratio=float(new.opt.impratio), timestep=float(new.opt.timestep),
                  note="Object, hand, friction, mass, reset and video controls unchanged; new GPU gate required")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "configure_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
