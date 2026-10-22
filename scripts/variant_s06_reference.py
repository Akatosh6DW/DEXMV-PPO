"""Create one documented S06 development variation without overwriting inputs."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from prepare_cylinder_s02_source import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=("aligned-wrist", "linear", "impratio", "timestep", "solref", "implicit"), required=True)
    p.add_argument("--value", type=float, default=2.2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    xml = ET.parse(args.input / "model.xml")
    with np.load(args.input / "reference.npz") as z:
        packet = dict(z)
    if args.kind == "aligned-wrist":
        initial = packet["initial_ctrl"].copy()
        initial[0] = packet["ctrl"][-1, 0]
        raw = packet["raw_ik_ctrl"]
        phase = packet["time"]/packet["time"][-1]
        ctrl = np.stack([np.interp(phase, np.linspace(0, 1, len(raw)), raw[:, k]) for k in range(17)], axis=1)
        packet["ctrl"] = initial + phase[:, None]*(ctrl-initial)
        packet["initial_ctrl"] = initial
    elif args.kind == "linear":
        phase = packet["time"]/packet["time"][-1]
        packet["ctrl"] = packet["initial_ctrl"] + phase[:, None]*(packet["ctrl"][-1]-packet["initial_ctrl"])
    elif args.kind == "implicit":
        xml.getroot().find("option").set("integrator", "implicit")
    elif args.kind == "solref":
        for geom in xml.getroot().iter("geom"):
            geom.set("solref", f"{args.value} 1")
    else:
        xml.getroot().find("option").set(args.kind, str(args.value))
    xml.write(args.output / "model.xml", encoding="unicode")
    np.savez_compressed(args.output / "reference.npz", **packet)
    (args.output / "variant.json").write_text(json.dumps(dict(
        input=str(args.input), kind=args.kind, value=args.value,
        source_model_sha256=digest(args.input / "model.xml"),
        source_reference_sha256=digest(args.input / "reference.npz")), indent=2)+"\n")


if __name__ == "__main__":
    main()
