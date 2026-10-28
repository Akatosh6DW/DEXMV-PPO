"""Portable S12 evaluation, weights-only continuation, or matched fresh training.

The original frozen job is unchanged. Only filesystem-dependent XML digests in
temporary runtime metadata are rebound, after checking identical XML semantics.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from restore_developer_release import ROOT, digest, contained, restore


def semantic_xml(payload, manifest, release):
    root = ET.fromstring(payload)
    known = {v["sha256"] for v in manifest["xml_assets"].values()}
    for node in root.iter():
        name = node.get("file")
        if name is None:
            continue
        if name in manifest["xml_assets"]:
            key = manifest["xml_assets"][name]["sha256"]
        else:
            path = Path(name)
            if not path.resolve().is_relative_to(release.resolve()):
                raise ValueError("Unexpected runtime asset outside release: " + name)
            key = digest(path)
            if key not in known:
                raise ValueError("Runtime mesh was changed")
        node.set("file", "sha256:" + key)
    # XML indentation is not part of the simulation definition.
    for node in root.iter():
        if node.text is not None and not node.text.strip():
            node.text = None
        node.tail = None
    return ET.tostring(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="guided_s25211",
                   choices=[f"{m}_s{s}" for m in ("ppo", "guided", "residual") for s in (25211,25212,25213)])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=0, help="Additional online steps; 0 evaluates only")
    p.add_argument("--from-scratch", action="store_true", help="Use original initialization and BC recipe")
    p.add_argument("--eval-count", type=int, default=8)
    p.add_argument("--eval-start", type=int, default=251010, help="Development seeds, never held-out tuning")
    p.add_argument("--evals", type=int, default=2)
    p.add_argument("--physics-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    a = p.parse_args()
    if a.steps < 0 or a.steps % 640 or a.eval_count < 1 or (a.steps and a.evals < 2):
        p.error("steps must be a nonnegative multiple of 640; eval-count >=1; training evals >=2")
    output = a.output.resolve()
    if output.exists() or output.is_relative_to(ROOT / "release") or output.is_relative_to(ROOT / "artifacts/orca_benchmark_suite/S12"):
        p.error("Use a new output directory outside frozen S12/release inputs")
    restore()
    release = ROOT / "release"
    manifest = json.loads((release / "developer_manifest.json").read_text())
    study = "artifacts/orca_benchmark_suite/S12/study_v2"

    def checked(relative):
        entry = manifest["files"][relative]
        path = contained(release, entry["storage"])
        if digest(path) != entry["sha256"]:
            raise ValueError("Release checksum mismatch: " + relative)
        return path

    protocol = json.loads(checked(study + "/protocol.json").read_text())
    for relative, expected in protocol["inputs"].items():
        if relative.startswith("scripts/") and digest(ROOT / relative) != expected:
            raise ValueError("Frozen source changed; use a new experiment protocol: " + relative)
    from s12_soft import SoftConfig, scene_xml
    cfg = SoftConfig(**protocol["config"])
    runtime_xml = scene_xml(cfg)
    original_xml = checked(study + "/model.xml").read_text()
    if semantic_xml(runtime_xml, manifest, release) != semantic_xml(original_xml, manifest, release):
        raise ValueError("Scene changed beyond asset path relocation")
    runtime_hash = hashlib.sha256(runtime_xml.encode()).hexdigest()
    config = json.loads(checked(study + f"/training/{a.case}/run_config.json").read_text())
    if config["model_sha256"] != digest(checked(study + "/model.xml")) or config["soft_config"] != asdict(cfg):
        raise ValueError("Policy was trained in a different scene")
    checkpoint = checked(study + f"/training/{a.case}/params.pkl")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = dict(case=a.case, mode="fresh_training" if a.from_scratch else
        "weights_only_warm_start" if a.steps else "evaluation", additional_steps=a.steps,
        original_steps=config["steps"], optimizer_restored=False, rng_restored=False,
        original_checkpoint_sha256=digest(checkpoint), original_model_sha256=config["model_sha256"],
        runtime_model_sha256=runtime_hash, scene_equivalence="XML identical after replacing asset paths by verified content hashes",
        protocol_sha256=digest(checked(study + "/protocol.json")), release_tag=manifest["tag"])
    with tempfile.TemporaryDirectory(prefix="s12-runtime-", dir=output.parent) as tmp:
        tmp = Path(tmp)
        config["model_sha256"] = runtime_hash
        (tmp / "run_config.json").write_text(json.dumps(config, indent=2)+"\n")
        (tmp / "params.pkl").symlink_to(checkpoint.resolve())
        argv = [sys.executable, str(ROOT / "scripts/run_s12_job.py"),
            "--method", config["method"], "--seed", str(config["seed"]),
            "--output", str(output), "--steps", str(a.steps), "--evals", str(a.evals),
            "--eval-start", str(a.eval_start), "--eval-count", str(a.eval_count),
            "--physics-workers", str(a.physics_workers), "--std", str(config["std"]),
            "--learning-rate", str(a.learning_rate), "--hand-offset-z", str(cfg.hand_offset_z)]
        if not a.from_scratch:
            argv += ["--checkpoint", str(tmp / "params.pkl")]
        elif config["method"] == "guided":
            folder = "artifacts/orca_benchmark_suite/S12/V2_D01_raise3mm"
            dataset = checked(folder + "/trajectories.npz")
            data_meta = json.loads(checked(folder + "/dataset_manifest.json").read_text())
            if data_meta["model_sha256"] != provenance["original_model_sha256"]:
                raise ValueError("Demonstration scene mismatch")
            data_meta["model_sha256"] = runtime_hash
            (tmp / "trajectories.npz").symlink_to(dataset.resolve())
            (tmp / "dataset_manifest.json").write_text(json.dumps(data_meta, indent=2)+"\n")
            argv += ["--dataset", str(tmp / "trajectories.npz"), "--bc-phase-only", "--bc-updates", "10000"]
        print(json.dumps(provenance, indent=2), flush=True)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
        result = subprocess.run(argv, cwd=ROOT, env=env)
        if output.is_dir():
            provenance["returncode"] = result.returncode
            (output / "continuation.json").write_text(json.dumps(provenance, indent=2)+"\n")
            # Preserve the input metadata after the temporary directory is removed.
            inputs = output / "initialization"
            inputs.mkdir(exist_ok=True)
            for path in tmp.iterdir():
                if path.is_symlink():
                    (inputs / path.name).symlink_to(os.path.relpath(path.resolve(), inputs))
                else:
                    shutil.copyfile(path, inputs / path.name)
            actual = json.loads((output / "run_config.json").read_text()) if (output / "run_config.json").exists() else None
            if actual:
                for key in ("checkpoint", "dataset"):
                    if actual.get(key):
                        actual[key] = str(inputs / Path(actual[key]).name)
                (output / "run_config.json").write_text(json.dumps(actual, indent=2)+"\n")
        result.check_returncode()


if __name__ == "__main__":
    main()
