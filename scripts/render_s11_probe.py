"""Render recorded cap physics without replacing it by the reference motion."""

import argparse
import json
from pathlib import Path

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(args.trace.parent / "model.xml"))
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 640
    data = mujoco.MjData(model)
    thread_ids = [i for i in range(model.neq) if model.equality(i).name.startswith("thread_")]
    cap = model.body("oakink_cap").id
    bottle = model.body("oakink_bottle").id
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .42, 135., -15.
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[[0, 1, 3]] = 1
    with np.load(args.trace) as trace, mujoco.Renderer(model, height=640, width=800) as renderer:
        with mediapy.VideoWriter(str(args.output / "cap_preload_release_then_loss.mp4"),
                                 shape=(640, 800), fps=20) as writer:
            for i, time in enumerate(trace["time"]):
                data.qpos[:] = trace["qpos"][i]
                data.qvel[:] = trace["qvel"][i]
                data.ctrl[:] = trace["ctrl"][i]
                data.mocap_pos[:] = trace["mocap_pos"][i]
                data.mocap_quat[:] = trace["mocap_quat"][i]
                data.eq_active[thread_ids] = not trace["released"][i]
                mujoco.mj_forward(model, data)
                if i == 0:
                    camera.lookat[:] = data.xipos[cap]
                renderer.update_scene(data, camera=camera, scene_option=option)
                if renderer.scene.ngeom < 10:
                    raise ValueError("Missing hand/object geometry in rendered scene")
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                label = (f"S11 P05 | t={time:.2f}s | turn={np.rad2deg(trace['angle'][i]):.1f}deg\n"
                         f"released={bool(trace['released'][i])} paired={bool(trace['paired'][i])} "
                         f"bottle_contact={bool(trace['bottle_contact'][i])}\n"
                         f"axial displacement={trace['lift'][i]*1000:.1f}mm | success=False")
                draw.rectangle((0, 0, 800, 52), fill=(20, 20, 20))
                draw.text((10, 5), label, fill="white")
                writer.add_image(np.asarray(frame))
                if i in (0, 199, 279, 319, len(trace["time"])-1):
                    frame.save(args.output / f"frame_{i:04d}.png")
        (args.output / "render.json").write_text(json.dumps(dict(
            trace=str(args.trace.resolve()), frames=len(trace["time"]), fps=20,
            kind="recorded mechanics probe, not a trained policy", camera="fixed close-up",
            cap_body=cap, bottle_body=bottle), indent=2)+"\n")


if __name__ == "__main__":
    main()
