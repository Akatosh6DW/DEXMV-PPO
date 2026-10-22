"""Predeclared, separate cap comparison; no best-checkpoint selection."""

import importlib.metadata
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from orca_formal_protocol import SOURCE_NAMES as CUBE_SOURCES
from prepare_hic_orca_object import digest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/orca_formal/CAP21_thread_stop_and_filter"
REFERENCE = ROOT / "artifacts/orca_formal/CAP10_sequence_mount_ik/kinematic_reference.npz"
DATASET = ROOT / "artifacts/orca_formal/CAP28_video_teacher_dataset/dataset.npz"
SEEDS = (401, 402, 403, 404, 405)
SOURCES = tuple(sorted(set(CUBE_SOURCES) | {
    "oakink_cap_protocol.py", "oakink_cap_env.py", "oakink_cap_native.py", "oakink_cap_dynamics.py",
    "train_oakink_cap.py", "smoke_oakink_cap.py", "audit_oakink_cap_run.py", "run_oakink_cap_formal.py"}))


def recipes():
    return [dict(label=f"s{seed}_{method}", seed=seed, method=method, steps=0 if method == "bc" else 655360)
            for seed in SEEDS for method in ("bc", "pure", "guided", "residual")]


def scenarios():
    return {"nominal": {}, "mass05": {"mass_scale": .5}, "mass15": {"mass_scale": 1.5},
            "friction07": {"friction_scale": .7}, "friction13": {"friction_scale": 1.3}}


def freeze(output, gate):
    inputs = [MODEL / "model.xml", MODEL / "report.json", REFERENCE, DATASET,
              DATASET.parent / "dataset_manifest.json", gate,
              ROOT / "artifacts/orca_formal/CAP32_source_provenance/manifest.json"]
    root = ET.parse(MODEL / "model.xml").getroot()
    for asset in root.findall("asset/*"):
        if asset.get("file"):
            path = Path(asset.get("file"))
            inputs.append(path if path.is_absolute() else MODEL / path)
    doc = ROOT / "docs/orca_formal_cap_protocol.md"
    inputs.append(doc)
    framework = ROOT / "Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src"
    inputs += [framework / name for name in ("mjx_env.py", "wrapper.py")]
    inputs += list((framework / "manipulation/orca_hand").glob("*.py"))
    packages = {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "mujoco", "brax", "flax", "optax", "numpy")}
    result = dict(scope="Separate source-matched fixed-bottle cap study, not exact physical screw teeth",
                  recipes=recipes(), seeds=list(SEEDS), evaluation_seeds=list(range(28500, 28564)),
                  monitoring_seeds=list(range(18400, 18416)), scenarios=scenarios(),
                  input_sha256={str(p.resolve()): digest(p) for p in inputs},
                  sources={name: digest(ROOT / "scripts" / name) for name in SOURCES}, packages=packages,
                  model=str(MODEL), reference=str(REFERENCE), dataset=str(DATASET), gate=str(gate.resolve()),
                  configuration=dict(tactile="on", num_envs=128, batch_size=128, evals=9, eval_count=16,
                                     eval_start=18400, bc_updates=2000, std=.7, learning_rate=.0003),
                  xla_flags=os.environ.get("XLA_FLAGS", ""), final_checkpoint_only=True, negative_results_retained=True)
    (output / "sources").mkdir()
    for name in SOURCES:
        (output / "sources" / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    (output / "protocol.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "protocol.md").write_bytes(doc.read_bytes())
    return result


def check_protocol(path):
    protocol = json.loads(Path(path).read_text())
    assert protocol["recipes"] == recipes()
    for filename, expected in protocol["input_sha256"].items():
        if digest(filename) != expected:
            raise ValueError("Frozen cap input changed: " + filename)
    for filename, expected in protocol["sources"].items():
        if digest(ROOT / "scripts" / filename) != expected:
            raise ValueError("Frozen cap source changed: " + filename)
    for name, expected in protocol["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Frozen cap package changed: " + name)
    return protocol


def check_recipe(args, protocol):
    recipe = next((r for r in protocol["recipes"] if r["seed"] == args.seed and r["method"] == args.method), None)
    if recipe is None or args.collect or args.steps != recipe["steps"]:
        raise ValueError("Not a predeclared cap training recipe")
    if args.output.name != recipe["label"] or os.environ.get("XLA_FLAGS", "") != protocol["xla_flags"]:
        raise ValueError("Formal cap run label/runtime differs")
    for key, expected in protocol["configuration"].items():
        if getattr(args, key) != expected:
            raise ValueError("Cap training configuration differs: " + key)
    for key in ("model", "reference", "gate"):
        if getattr(args, key).resolve() != Path(protocol[key]).resolve():
            raise ValueError("Cap training input differs: " + key)
    if args.method == "bc" and args.dataset.resolve() != Path(protocol["dataset"]).resolve():
        raise ValueError("Different cap demonstration data")
    if args.method == "guided":
        expected = args.output.parent / f"s{args.seed}_bc/bc_params.pkl"
        if args.bc_checkpoint.resolve() != expected.resolve():
            raise ValueError("Guided cap initialization is not the paired run")
