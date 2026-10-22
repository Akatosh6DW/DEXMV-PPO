"""Save geometry-only transfer gates and initial palm-view images before testing."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Aero-Hand/sim_rl/mujoco_playground"))

import jax
import mujoco
import numpy as np
from PIL import Image
from inspect_orca_mesh_alignment import build_overlay
from orca_s03_objects import OrcaS03ObjectGrasp, SHAPES
from train_orca_acquisition import MODEL, REFERENCE, initial_gate
from prepare_hic_orca_object import digest


def main():
    out = ROOT / "artifacts/orca_benchmark_suite/S03/object_geometry"
    out.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_pmap_shmap_merge", False)
    records = {}
    for shape in SHAPES:
        directory = out / shape
        directory.mkdir()
        env = OrcaS03ObjectGrasp(MODEL, "residual", REFERENCE, shape=shape)
        gate = initial_gate(env)
        env.export_model(directory / "model.xml")
        state = jax.device_get(jax.jit(env.reset)(jax.random.PRNGKey(60050)))
        model, overlay = build_overlay(directory / "model.xml", directory / "visual_overlay.xml")
        data = mujoco.MjData(model)
        data.qpos[:] = state.data.qpos
        data.mocap_pos[:] = state.data.mocap_pos
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.035, .025, .245]
        camera.distance, camera.azimuth, camera.elevation = .38, 225, -18
        option = mujoco.MjvOption()
        pixels = {}
        with mujoco.Renderer(model, 480, 640) as renderer:
            for mode in ("stl", "collision"):
                option.geomgroup[:] = 1
                option.geomgroup[2] = mode == "stl"
                option.geomgroup[3] = mode == "collision"
                renderer.update_scene(data, camera, scene_option=option)
                frame = renderer.render().copy()
                if frame.std() < 5:
                    raise AssertionError("Blank geometry preview")
                Image.fromarray(frame).save(directory / (mode+".png"))
                pixels[mode] = float(frame.std())
        bid = env.mj_model.body("cube").id
        records[shape] = dict(initial_gate=gate, half_height_m=env.half_height,
                              mass_kg=float(env.mj_model.body_mass[bid]),
                              principal_inertia_kg_m2=env.mj_model.body_inertia[bid].tolist(),
                              shape=SHAPES[shape], initial_state_seed=60050,
                              model_sha256=digest(directory / "model.xml"),
                              source_model_sha256=digest(MODEL), reference_sha256=digest(REFERENCE),
                              pixels_std=pixels, exact_hand_control_contract=True)
    result = dict(passed=True, policy_performance_not_used=True, shapes=records,
                  source_sha256={name: digest(ROOT / "scripts" / name)
                                 for name in ("orca_s03_objects.py", "prepare_orca_s03_objects.py")})
    (out / "gates.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
