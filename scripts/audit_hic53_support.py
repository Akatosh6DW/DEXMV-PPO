"""Isolate object/support contact from hand retargeting in native MuJoCo and MJX."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import jax
import mujoco
from mujoco import mjx
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(args.model)
    root = tree.getroot()
    world = root.find("worldbody")
    for body in list(world.findall("body")):
        if body.get("name") not in ("cube", "cube_support"):
            world.remove(body)
    root.remove(root.find("actuator"))
    for mesh in root.findall("asset/mesh"):
        path = Path(mesh.get("file"))
        if not path.is_absolute():
            mesh.set("file", str((args.model.parent / path).resolve()))
    path = args.output / "isolated.xml"
    tree.write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    initial = data.qpos.copy()
    steps = round(2. / model.opt.timestep)
    native = []
    for _ in range(steps):
        mujoco.mj_step(model, data)
        native.append(data.qpos.copy())
    assert jax.default_backend() == "gpu"
    jax.config.update("jax_default_matmul_precision", "highest")
    device_model = mjx.put_model(model)
    def run(data):
        def step(data, _):
            nxt = mjx.step(device_model, data)
            return nxt, nxt.qpos
        return jax.lax.scan(step, data, None, length=steps)[1]
    gpu = np.asarray(jax.jit(run)(mjx.make_data(model)))
    native = np.asarray(native)
    result = dict(backend=jax.default_backend(), seconds=2.,
                  native_max_center_displacement_m=float(np.linalg.norm(native[:, :3] - initial[:3], axis=1).max()),
                  mjx_max_center_displacement_m=float(np.linalg.norm(gpu[:, :3] - initial[:3], axis=1).max()),
                  initial_qpos=initial.tolist(), native_final=native[-1].tolist(), mjx_final=gpu[-1].tolist())
    result["support_pass"] = max(result["native_max_center_displacement_m"], result["mjx_max_center_displacement_m"]) < .002
    np.savez_compressed(args.output / "trace.npz", native=native, mjx=gpu, initial=initial)
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
