"""Run a frozen S10 policy from release assets, without the local artifacts tree."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT / "release")
    parser.add_argument("--case", default="guided_s22101")
    parser.add_argument("--seed", type=int, default=220000)
    parser.add_argument("--steps", type=int, default=840)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 840:
        parser.error("steps must be in [1, 840]")
    release = args.release.resolve()
    manifest = json.loads((release / "manifest.json").read_text())
    matches = [e for e in manifest["inventory"]["evaluations"]
               if e["group"] == "S10" and Path(e["config"]).parent.name == args.case]
    if len(matches) != 1:
        parser.error("case must identify one S10 evaluation")
    entry = matches[0]

    def checked(relative):
        path = release / relative
        expected = manifest["files"][relative]["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Release integrity failure: " + relative)
        return path

    record = manifest["records"][entry["config"]]
    config = json.loads(checked(record["config"]).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    artifact_root = (ROOT / "artifacts").resolve()
    output_root = args.output.resolve()

    def forbid_history_read(event, values):
        if event == "open" and isinstance(values[0], (str, bytes)):
            path = Path(values[0].decode() if isinstance(values[0], bytes) else values[0]).resolve()
            if path.is_relative_to(artifact_root) and not path.is_relative_to(output_root):
                raise PermissionError("Portable replay attempted to open local experiment history: " + str(path))

    sys.addaudithook(forbid_history_read)
    with tempfile.TemporaryDirectory(prefix="dexmv-release-") as temporary:
        work = Path(temporary)
        source = work / "scripts"
        source.mkdir()
        for name, relative in record["sources"].items():
            shutil.copyfile(checked(relative), source / name)
        inputs = work / "inputs"
        inputs.mkdir()
        shutil.copyfile(checked(manifest["blobs"][entry["inputs"]["reference"]]), inputs / "reference.npz")
        selection = json.loads(checked(manifest["blobs"][entry["inputs"]["selection"]]).read_text())
        scene = manifest["portable_scenes"][entry["inputs"]["model"]]
        original = checked(scene["file"])
        tree = ET.parse(original)
        for node in tree.getroot().iter():
            if "file" in node.attrib:
                asset = (original.parent / node.get("file")).resolve()
                checked(str(asset.relative_to(release)))
                node.set("file", str(asset))
        model = inputs / "model.xml"
        tree.write(model, encoding="unicode")
        sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))
        sys.path.insert(0, str(source))
        import jax
        import numpy as np
        from brax.training.agents.ppo import networks
        import s10_mobile_base
        from orca_s10_env import OrcaS10Grasp
        from train_orca_matched import initialize_matched

        jax.config.update("jax_default_matmul_precision", "highest")
        s10_mobile_base.INPUT = inputs
        arm = "residual" if entry["method"] == "residual" else "pure"
        env = OrcaS10Grasp(model, selection, arm)
        net, params = initialize_matched(env, config["seed"], config["std"])
        if entry["policy_sha256"]:
            params = pickle.loads(checked(manifest["blobs"][entry["policy_sha256"]]).read_bytes())
        inference = jax.jit(networks.make_inference_fn(net)(params, deterministic=True))
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        rng = jax.random.PRNGKey(args.seed)
        state = reset(rng)
        trace = []
        for index in range(args.steps):
            rng, key = jax.random.split(rng)
            action = (env.ctrl_to_action(env.reference_base(state.info)) if entry["method"] == "reference"
                      else inference(state.obs, key)[0])
            state = step(state, action)
            row = jax.device_get(dict(qpos=state.data.qpos, ctrl=state.data.ctrl, action=action,
                                      hold=state.metrics["hold_sec"], best=state.metrics["best_hold_sec"],
                                      success30=state.metrics["success30"], done=state.done))
            if not all(np.isfinite(v).all() for v in row.values()):
                raise FloatingPointError("Nonfinite policy rollout")
            trace.append(row)
            if row["done"]:
                break
        arrays = {k: np.asarray([r[k] for r in trace]) for k in trace[0]}
        np.savez_compressed(args.output / "trace.npz", **arrays)
        report = dict(case=args.case, seed=args.seed, steps=len(trace), backend=jax.default_backend(),
                      policy_sha256=entry["policy_sha256"], final_hold_sec=float(trace[-1]["hold"]),
                      best_hold_sec=float(trace[-1]["best"]), success30=bool(trace[-1]["success30"]),
                      done=bool(trace[-1]["done"]), inputs="release only, no artifacts reads required",
                      python_history_open_guard=True,
                      scope="single-case portable XML inference check, not a repetition of the GPU batch benchmark")
        (args.output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
