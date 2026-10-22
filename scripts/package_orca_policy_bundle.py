"""Archive the three S04 policy groups and relocate their shared scene assets."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "deliverables/Orca_Video_PPO_S04"
MODEL = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
REFERENCE = ROOT / "artifacts/orca_cube_acquisition/E01_flex50_refinement/candidate_reference.npz"
BASE = ROOT / "artifacts/orca_benchmark_suite"
GROUPS = {"pure_ppo": "ppo_tuned", "bc_ppo": "guided_tuned", "residual_ppo": "residual_tuned"}
BRIDGES = (
    "prepare_hic_cube_demo.py", "retarget_hic_cube_to_dexmv.py",
    "prepare_hic_orca_object.py", "probe_hic53_orca.py",
    "probe_orca_cube_acquisition.py", "train_orca_acquisition.py",
    "orca_residual_controls.py", "orca_residual_env.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def script_dependencies(names):
    pending, seen = list(names), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        tree = ast.parse((ROOT / "scripts" / name).read_text())
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.level == 0
                       else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                leaf = (module or "").split(".")[0] + ".py"
                if (ROOT / "scripts" / leaf).is_file():
                    pending.append(leaf)
    return sorted(seen)


def relocate_scene(source, destination, copy):
    tree = ET.parse(source)
    root = tree.getroot()
    if root.findall(".//include"):
        raise ValueError("Resolve includes explicitly before packaging this scene")
    compiler = root.find("compiler")
    for node in root.iter():
        if "file" not in node.attrib:
            continue
        path = Path(node.attrib["file"])
        if not path.is_absolute():
            key = "meshdir" if node.tag == "mesh" else "texturedir" if node.tag == "texture" else "assetdir"
            directory = compiler.get(key, compiler.get("assetdir", "")) if compiler is not None else ""
            path = (source.parent / directory / path).resolve()
        asset = Path("assets") / (sha(path)[:16] + "_" + path.name)
        copy(path, destination.parent / asset)
        node.set("file", asset.as_posix())
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
    ET.indent(tree)
    tree.write(destination, encoding="unicode")


def compare_models(source, moved):
    old, new = (mujoco.MjModel.from_xml_path(str(p)) for p in (source, moved))
    arrays = 0
    # Asset path byte buffers and their offsets necessarily change on relocation.
    for name in dir(old):
        if name.startswith("_") or "path" in name:
            continue
        value = getattr(old, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(new, name), err_msg=name)
            arrays += 1
    for name in dir(old.opt):
        if name.startswith("_"):
            continue
        value = getattr(old.opt, name)
        if isinstance(value, (int, float, np.ndarray)):
            np.testing.assert_array_equal(value, getattr(new.opt, name), err_msg="opt." + name)
    return dict(compiled_arrays_equal=arrays, nq=new.nq, nv=new.nv, nu=new.nu,
                timestep=new.opt.timestep, excluded="asset path buffers and offsets only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs, copied = {}, {}

    def copy(source, target):
        source = source.resolve()
        original_hash = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != original_hash:
            raise ValueError("Copy failed: " + str(target))
        inputs[str(source.relative_to(ROOT))] = original_hash
        copied[str(target.relative_to(out))] = str(source.relative_to(ROOT))

    protocol_path = BASE / "S04/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    copy(protocol_path, out / "provenance/S04_protocol.original.json")
    policy_index, runtime = [], set()
    for method, group in GROUPS.items():
        for seed in range(21001, 21011):
            run = BASE / "S03/confirmation" / group / f"s{seed}"
            checkpoint = run / "step_000163840/params.pkl"
            key = str(checkpoint.relative_to(ROOT))
            if sha(checkpoint) != protocol["source_hashes"][key]:
                raise ValueError("Not the frozen S04 checkpoint: " + key)
            config = json.loads((run / "run_config.json").read_text())
            if config["model_sha256"] != sha(MODEL) or config["reference_sha256"] != sha(REFERENCE):
                raise ValueError("Checkpoint model/reference mismatch")
            dest = out / "weights" / method / f"seed_{seed}"
            copy(checkpoint, dest / "params.pkl")
            for name in ("run_config.json", "env_config.json", "metrics.csv"):
                copy(run / name, dest / name)
            if method == "bc_ppo":
                copy(run / "bc_params.pkl", dest / "bc_initial_params.pkl")
            for name, expected in config["source_sha256"].items():
                path = run / "sources" / name
                if sha(path) != expected:
                    raise ValueError("Training source hash mismatch")
                if name in runtime:
                    if sha(out / "runtime_sources" / name) != expected:
                        raise ValueError("Different source versions need separate folders")
                else:
                    copy(path, out / "runtime_sources" / name)
                    runtime.add(name)
            policy_index.append(dict(method=method, seed=seed, checkpoint_steps=163840,
                                     original_training_total_steps=config["steps"],
                                     checkpoint=str((dest / "params.pkl").relative_to(out)),
                                     checkpoint_sha256=sha(checkpoint),
                                     action_semantics="reference_plus_residual" if method == "residual_ppo" else "absolute_target",
                                     model="scene/physics/cube.xml", reference="reference/candidate_reference.npz",
                                     original_model_sha256=sha(MODEL), original_reference_sha256=sha(REFERENCE)))
    for name in script_dependencies(BRIDGES):
        copy(ROOT / "scripts" / name, out / "bridge/scripts" / name)
    for name, source in {
        "candidate_reference.npz": REFERENCE,
        "unrefined_reference.npz": MODEL.parent / "reference.npz",
        "refinement_report.json": REFERENCE.parent / "report.json",
    }.items():
        copy(source, out / "reference" / name)
    scenes = {"physics": MODEL,
              "visual": BASE / "S04/videos/residual/visual_overlay.xml"}
    checks = {}
    for name, source in scenes.items():
        copy(source, out / "provenance" / (name+"_scene.original.xml"))
        target = out / "scene" / name / "cube.xml"
        target.parent.mkdir(parents=True, exist_ok=True)
        relocate_scene(source, target, copy)
        checks[name] = compare_models(source, target)
    # Load the relocated copies from an unrelated directory, not just the original tree.
    with tempfile.TemporaryDirectory(prefix="orca-bundle-") as tmp:
        shutil.copytree(out / "scene", Path(tmp) / "scene")
        for name, source in scenes.items():
            compare_models(source, Path(tmp) / "scene" / name / "cube.xml")
    with np.load(out / "reference/candidate_reference.npz", allow_pickle=False) as z:
        if z["ctrl"].shape != (801, 17) or z["time"].shape != (801,):
            raise ValueError("Unexpected reference shape")
        reference = dict(ctrl_shape=list(z["ctrl"].shape), time_start_sec=float(z["time"][0]),
                         time_end_sec=float(z["time"][-1]), source_frames=[30, 95],
                         description="Offline corrected joint targets, not neural network weights")
    (out / "weights/index.json").write_text(json.dumps(policy_index, indent=2, ensure_ascii=False)+"\n")
    (out / "reference/contents.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False)+"\n")
    (out / "verification.json").write_text(json.dumps(dict(scenes=checks, relocated_load=True,
        policy_count=len(policy_index), frozen_checkpoints_verified=True, no_policy_rollouts_run=True), indent=2)+"\n")
    for name, expected in inputs.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Input changed during packaging")
    manifest = dict(version="S04-163840-all10seeds", created_date="2026-10-09",
                    copied_from=copied, input_sha256=inputs,
                    output_sha256={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
                    policy_index="weights/index.json", shared_scene=True,
                    standalone_training_package=False, package_script_sha256=sha(Path(__file__)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(output=str(out), policies=len(policy_index), bridge_scripts=len(script_dependencies(BRIDGES)),
                          scene_checks=checks, relocated_load=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
