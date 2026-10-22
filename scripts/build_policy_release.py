"""Package evaluated policies, exact sources, inputs and compact result tables."""

import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from inventory_policy_release import ROOT, digest, eval_roots
from package_orca_policy_bundle import relocate_scene


def verify_scene(source, destination):
    original = mujoco.MjModel.from_xml_path(str(source))
    moved = mujoco.MjModel.from_xml_path(str(destination))
    differences = {}
    exact = 0
    for name in dir(original):
        if name.startswith("_") or "path" in name:
            continue
        a, b = getattr(original, name), getattr(moved, name)
        if not isinstance(a, np.ndarray):
            continue
        if np.array_equal(a, b):
            exact += 1
        elif name == "mesh_polynormal":
            np.testing.assert_allclose(a, b, atol=1e-10, rtol=0, err_msg=name)
            differences[name] = dict(max_abs=float(abs(a-b).max()), tolerance=1e-10)
        elif name == "mesh_polyvert":
            # Convex hull polygons can start at a different vertex after recompilation.
            for start, count in zip(original.mesh_polyvertadr, original.mesh_polyvertnum):
                first, second = a[start:start+count], b[start:start+count]
                if not any(np.array_equal(first, np.roll(second, shift)) for shift in range(count)):
                    raise ValueError("Relocated polygon changed its ordered boundary")
            differences[name] = "cyclic vertex-order equivalence checked for every polygon"
        else:
            np.testing.assert_array_equal(a, b, err_msg=name)
    for name in dir(original.opt):
        if name.startswith("_"):
            continue
        a = getattr(original.opt, name)
        if isinstance(a, (int, float, np.ndarray)):
            np.testing.assert_array_equal(a, getattr(moved.opt, name), err_msg="opt."+name)
    binary = destination.with_suffix(".mjb")
    mujoco.mj_saveModel(original, str(binary))
    restored = mujoco.MjModel.from_binary_path(str(binary))
    count = 0
    for name in dir(original):
        if name.startswith("_"):
            continue
        a = getattr(original, name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, getattr(restored, name), err_msg="binary."+name)
            count += 1
    return dict(xml_exact_arrays=exact, xml_equivalent_arrays=differences,
                binary_exact_arrays=count, mujoco_version=mujoco.__version__,
                nq=original.nq, nv=original.nv, nu=original.nu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    if inventory["errors"]:
        raise ValueError("Resolve all inventory errors before packaging")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    files, records, blobs, scenes = {}, {}, {}, {}

    def copy(source, target):
        key = digest(source)
        relative = str(target.relative_to(output))
        if target.exists():
            if digest(target) != key:
                raise ValueError("Conflicting package path: " + relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        files[relative] = dict(sha256=key, bytes=source.stat().st_size)
        return relative

    def blob(source):
        key = digest(source)
        destination = output / "blobs" / (key + source.suffix)
        relative = copy(source, destination)
        blobs[key] = relative
        return relative

    def record_config(path):
        relative = str(path.relative_to(ROOT))
        if relative in records:
            return
        config = json.loads(path.read_text())
        record = dict(config=copy(path, output / "records" / relative), sources={})
        for name in ("summary.json", "metrics.csv", "env_config.json", "bc_loss.json", "ppo_metrics.jsonl"):
            source = path.parent / name
            if source.is_file():
                copy(source, output / "records" / source.relative_to(ROOT))
        for name, expected in config.get("source_sha256", config.get("sources", {})).items():
            if expected in blobs:
                record["sources"][name] = blobs[expected]
                continue
            source = path.parent / "sources" / name
            if not source.is_file() or digest(source) != expected:
                candidate = ROOT / "scripts" / name
                if candidate.is_file() and digest(candidate) == expected:
                    source = candidate
                else:
                    raise ValueError("Missing or changed frozen source: " + str(source))
            record["sources"][name] = blob(source)
        records[relative] = record

    for key, item in inventory["policies"].items():
        source = ROOT / item["paths"][0]
        if digest(source) != key:
            raise ValueError("Policy changed since inventory")
        blob(source)
        for original in item["paths"]:
            folder = (ROOT / original).parent
            if not (folder / "run_config.json").is_file():
                folder = folder.parent
            record_config(folder / "run_config.json")
    for item in inventory["evaluations"]:
        record_config(ROOT / item["config"])
    for key, item in inventory["inputs"].items():
        source = ROOT / item["paths"][0]
        if digest(source) != key:
            raise ValueError("Input changed since inventory")
        blob(source)
        if source.suffix == ".xml":
            destination = output / "scenes" / key / "model.xml"
            destination.parent.mkdir(parents=True, exist_ok=args.resume)
            relocate_scene(source, destination, copy)
            verification = verify_scene(source, destination)
            relative = str(destination.relative_to(output))
            files[relative] = dict(sha256=digest(destination), bytes=destination.stat().st_size)
            binary = destination.with_suffix(".mjb")
            binary_relative = str(binary.relative_to(output))
            files[binary_relative] = dict(sha256=digest(binary), bytes=binary.stat().st_size)
            scenes[key] = dict(file=relative, binary=binary_relative, verification=verification)
    for label, folder in eval_roots().items():
        base = folder.parent
        for name in ("protocol.json", "statistics_plan.json", "benchmark_card.md", "benchmark_card.json"):
            source = base / name
            if source.is_file():
                copy(source, output / "records" / source.relative_to(ROOT))
        if (base / "report").is_dir():
            for source in (base / "report").rglob("*"):
                if source.is_file() and source.suffix in (".csv", ".json", ".md", ".png"):
                    copy(source, output / "records" / source.relative_to(ROOT))
    # Keep S11 evidence with the publication; there is no learned S11 policy.
    s11 = ROOT / "artifacts/orca_benchmark_suite/S11"
    for folder in sorted(s11.glob("P*")):
        for name in ("summary.json", "episodes.json", "fit.json", "fit_s11_grip.py", "s11_cap.py"):
            source = folder / name
            if source.is_file():
                copy(source, output / "records" / source.relative_to(ROOT))
    manifest = dict(schema=1, inventory=inventory, records=records, blobs=blobs,
                    portable_scenes=scenes, files=files,
                    exclusions="Raw source videos, training rollout datasets and redundant development checkpoints stay local")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for relative, entry in files.items():
        if digest(output / relative) != entry["sha256"]:
            raise ValueError("Final package hash failure: " + relative)
    summary = dict(status="packaged_and_hash_verified", policies=len(inventory["policies"]),
                   scenes=len(scenes), files=len(files), bytes=sum(v["bytes"] for v in files.values()),
                   manifest_sha256=digest(output / "manifest.json"))
    (output / "build.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
