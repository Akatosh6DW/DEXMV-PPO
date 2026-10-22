"""Check the matched object/reference pair before enabling residual training."""

import argparse
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from prepare_hic_orca_object import digest, MODEL, SOURCE
from probe_hic53_orca import native_rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    old = mujoco.MjModel.from_xml_path(str(MODEL))
    mount_path = args.model.parent / "mount_transform.json"
    mount_check = None
    if mount_path.exists():
        from rotate_orca_mount import verify_transform
        mount_check = verify_transform(args.model, args.reference, mount_path)
        proof = json.loads(mount_path.read_text())
        parent = mujoco.MjModel.from_xml_path(proof["parent_model"])
        # Only a proved change to the fixed mount is exempt from the G08 pose.
        for field in ("body_pos", "body_quat"):
            np.testing.assert_array_equal(getattr(parent, field)[parent.body("right_mount").id],
                                          getattr(old, field)[old.body("right_mount").id])
    for i in range(old.nbody):
        if old.body(i).name in ("cube", "cube_support"):
            continue
        for field in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat"):
            if mount_check and old.body(i).name == "right_mount" and field in ("body_pos", "body_quat", "body_ipos", "body_iquat"):
                continue
            np.testing.assert_array_equal(getattr(model, field)[i], getattr(old, field)[i])
    for i in range(old.ngeom):
        if old.geom(i).name == "cube":
            continue
        new_id = model.geom(old.geom(i).name).id
        for field in ("geom_size", "geom_pos", "geom_quat", "geom_type", "geom_friction", "geom_contype", "geom_conaffinity"):
            np.testing.assert_array_equal(getattr(model, field)[new_id], getattr(old, field)[i])
    for field in ("jnt_range", "jnt_axis", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm"):
        np.testing.assert_array_equal(getattr(model, field), getattr(old, field))
    with np.load(args.reference) as z:
        fields = {k: z[k] for k in z.files}
    # Match the training loader's (step+1)*dt sampling, not the old t=0 replay.
    from orca_residual_controls import load_reference
    base, _ = load_reference(args.reference, *model.actuator_ctrlrange.T,
                             fields["initial_ctrl"], .05, 800, .1)
    checks = []
    for seed in range(8):
        trace, check = native_rollout(model, fields["object_position"], fields["object_quaternion"],
                                      base, seed, initial_ctrl=fields["initial_ctrl"])
        checks.append(check)
        if seed == 0:
            np.savez_compressed(args.output / "native_trace.npz", **trace)
    ok = all(c["reset_penetration_m"] <= .003 and c["max_penetration_m"] <= .005
             and c["max_abs_qvel"] < 1000 for c in checks)
    paths = [MODEL, SOURCE / "models_motion/20/Model_Object_Cube.OFF",
             args.reference.parent / "report.json"]
    if mount_check:
        paths += [mount_path, Path(proof["parent_model"]), Path(proof["parent_reference"])]
    proxy_path = args.model.parent / "object_proxy_report.json"
    if proxy_path.exists():
        proxy = json.loads(proxy_path.read_text())
        assert proxy["model_sha256"] == digest(args.model)
        assert proxy["source_sha256"] == digest(SOURCE / "models_motion/20/Model_Object_Cube.OFF")
        assert proxy["accepted"]
        if proxy["mode"] == "box":
            assert proxy["box_corner_to_source_max_mm"] <= 2.
        else:
            assert proxy["bidirectional_max_sampled_mm"] <= 2.
        paths.append(proxy_path)
    for mesh in ET.parse(args.model).findall("asset/mesh"):
        path = Path(mesh.get("file"))
        paths.append(path if path.is_absolute() else args.model.parent / path)
    result = dict(geometry_and_native_pass=ok, pass_mjx=False, backend="not_run",
                  model=str(args.model.resolve()), reference=str(args.reference.resolve()),
                  model_sha256=digest(args.model), reference_sha256=digest(args.reference),
                  asset_sha256={str(p.resolve()): digest(p) for p in paths}, native_checks=checks,
                  mount_transform_check=mount_check,
                  xla_flags=os.environ.get("XLA_FLAGS", ""),
                  note="Geometry and stability gate, not a performance pass; same 30s clean-hold criterion")
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print("NATIVE", json.dumps(dict(pass_native=ok, checks=checks)), flush=True)
    if not ok:
        raise SystemExit("Native gate failed; do not train")
    if args.gpu:
        import jax
        from orca_hic53_env import OrcaHic53Grasp
        from orca_guidance_recheck import score
        from train_orca_refit import evaluator
        from train_orca_residual import initialize_zero_mean
        assert jax.default_backend() == "gpu"
        jax.config.update("jax_default_matmul_precision", "highest")
        jax.config.update("jax_compilation_cache_dir", str(Path("artifacts/orca30/jax_cache").resolve()))
        env = OrcaHic53Grasp(args.model, "residual", args.reference)
        net, params = initialize_zero_mean(101, .7)
        started = time.monotonic()
        trace = jax.device_get(evaluator(env, net, [9100, 9101])(params))
        assert all(np.isfinite(v).all() for v in trace.values())
        row, holds, success = score(trace, env)
        np.savez_compressed(args.output / "mjx_trace.npz", **trace)
        acquisition_pass = bool(np.all(trace["alive"][:, 39]))
        result.update(pass_mjx=acquisition_pass, mjx_acquisition_pass=acquisition_pass,
                      mjx_active_steps=trace["active"].sum(axis=1).tolist(),
                      backend=jax.default_backend(), mjx_elapsed_sec=time.monotonic() - started,
                      mjx_metrics=row, mjx_holds=holds.tolist(), mjx_success=success.tolist())
        (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print("MJX", json.dumps(row), flush=True)
        if not acquisition_pass:
            raise SystemExit("Object dropped before support withdrawal; acquisition gate failed")


if __name__ == "__main__":
    main()
