"""Check package bytes, trusted policy tensors, input arrays and portable scenes."""

import argparse
import json
from pathlib import Path
import pickle

import jax
import mujoco
import numpy as np

from inventory_policy_release import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path("release"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.release.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, entry in manifest["files"].items():
        path = root / relative
        if not path.is_relative_to(root) or digest(path) != entry["sha256"]:
            raise ValueError("File mismatch: " + relative)
    policies = []
    for key in manifest["inventory"]["policies"]:
        path = root / manifest["blobs"][key]
        # Only run this verifier on the locally built / otherwise trusted release.
        params = pickle.loads(path.read_bytes())
        leaves = jax.tree_util.tree_leaves(params)
        numbers = 0
        for leaf in leaves:
            value = np.asarray(leaf)
            if value.dtype.kind in "biufc":
                if not np.isfinite(value).all():
                    raise ValueError("Nonfinite policy parameter: " + key)
                numbers += value.size
        if not numbers:
            raise ValueError("No policy parameters found: " + key)
        policies.append(dict(sha256=key, numeric_parameters=numbers))
    arrays = 0
    for key in manifest["inventory"]["inputs"]:
        path = root / manifest["blobs"][key]
        if path.suffix != ".npz":
            continue
        with np.load(path, allow_pickle=False) as packet:
            for name in packet.files:
                value = packet[name]
                if value.dtype.kind in "biufc" and not np.isfinite(value).all():
                    raise ValueError("Nonfinite reference array: " + key + ":" + name)
                arrays += 1
    scenes = []
    for key, entry in manifest["portable_scenes"].items():
        xml = mujoco.MjModel.from_xml_path(str(root / entry["file"]))
        binary = mujoco.MjModel.from_binary_path(str(root / entry["binary"]))
        if (xml.nq, xml.nv, xml.nu) != (binary.nq, binary.nv, binary.nu):
            raise ValueError("Scene layout differs")
        for model in (xml, binary):
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or np.any(data.warning.number):
                raise ValueError("Invalid scene initialization: " + key)
        scenes.append(dict(original_sha256=key, nq=binary.nq, nu=binary.nu))
    report = dict(status="passed", manifest_sha256=digest(root / "manifest.json"),
                  files=len(manifest["files"]), policies=policies, reference_arrays=arrays, scenes=scenes,
                  scope="Integrity, deserialization, finite tensors and scene initialization; not policy performance evaluation")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(status="passed", files=report["files"], policies=len(policies),
                         reference_arrays=arrays, scenes=len(scenes)), indent=2))


if __name__ == "__main__":
    main()
