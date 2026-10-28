"""Render actual S12 soft vertices, not the rigid reference or a fixed shell."""

import argparse
import json
from pathlib import Path

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def load_snapshot(model, data, trace, index):
    """Match cached geometry from before the last implicit position integration."""
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICIT:
        raise ValueError("Snapshot reconstruction is validated for the S12 implicit integrator only")
    data.qpos[:] = trace["qpos"][index]
    data.qvel[:] = trace["qvel"][index]
    data.ctrl[:] = trace["ctrl"][index]
    data.mocap_pos[:] = trace["mocap"][index]
    data.mocap_quat[:] = trace["mocap_quat"][index]
    # mj_step advances qpos after computing flexvert_xpos. Rewind the position
    # update, using its final velocity, so every rendered vertex matches the log.
    mujoco.mj_integratePos(model, data.qpos, data.qvel, -model.opt.timestep)
    data.time = float(trace["time"][index]) - model.opt.timestep
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.flexvert_xpos, trace["vertices"][index], atol=5e-5)
    return float(np.max(np.abs(data.flexvert_xpos - trace["vertices"][index])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=24.)
    p.add_argument("--model", type=Path)
    p.add_argument("--label", default="reference")
    p.add_argument("--mode", choices=("visual", "collision"), default="visual")
    p.add_argument("--short", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    model_path = a.model or a.trace.parent/"model.xml"
    if not model_path.exists():
        model_path = a.trace.parent.parent/"model.xml"
    if a.mode == "visual":
        from inspect_orca_mesh_alignment import build_overlay
        model, _ = build_overlay(model_path, a.output/"visual_overlay.xml")
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 640
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.055, .04, .50]
    camera.azimuth, camera.elevation, camera.distance = 225., -18., .45
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[[0, 1, 2 if a.mode == "visual" else 3]] = 1
    option.flexgroup[:] = 1
    # Drawing both FLEXFACE and FLEXSKIN produces overlapping surfaces.
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = a.mode == "collision"
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = a.mode == "visual"
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = a.mode == "collision"
    maximum_vertex_error = 0.
    with np.load(a.trace) as trace, mujoco.Renderer(model, height=640, width=800) as renderer:
        count = min(len(trace["time"]), round(a.seconds/.05))
        success = bool(trace["success15"][-1])
        stop_reason = "requested_duration"
        if a.short:
            if success:
                count = min(count, int(np.flatnonzero(trace["hold_sec"]>=4)[0])+1)
                stop_reason = "four_seconds_stable_hold; full_episode_verified_separately"
            else:
                quality = ((trace["forbidden_force_n"]>=.02)|(trace["max_principal_strain"]>.3)
                           |(trace["penetration_m"]>.003)|(trace["peak_force_n"]>30))
                dropped = ~trace["alive"].astype(bool)
                collapsed = trace["min_volume_ratio"]<=.05
                impossible = (24-trace["time"]+trace["hold_sec"])<15-1e-6
                bad = ((trace["time"]>=5.5)&quality)|dropped|impossible
                if bad.any():
                    first = int(np.flatnonzero(bad)[0])
                    count = min(count, first+1)
                    stop_reason = ("element_collapse_or_inversion" if collapsed[first] else
                                   "drop" if dropped[first] else "quality_violation" if quality[first]
                                   else "insufficient_time_remaining_for_15s_hold")
        video = f"S12_soft_cylinder_{a.label}_palm_{a.mode}_{'success' if success else 'failure'}.mp4"
        with mediapy.VideoWriter(str(a.output/video),
                                 shape=(640, 800), fps=20) as writer:
            for i in range(count):
                maximum_vertex_error = max(maximum_vertex_error, load_snapshot(model, data, trace, i))
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 800, 54), fill=(20, 20, 20))
                text = (f"S12 | soft cylinder | {a.label} | t={trace['time'][i]:.2f}s | full test={'PASS' if success else 'FAIL'}\n"
                        f"strain={100*trace['max_principal_strain'][i]:.1f}% "
                        f"penetration={1000*trace['penetration_m'][i]:.2f}mm "
                        f"hold={trace['hold_sec'][i]:.2f}s\n"
                        f"tray force={trace['support_force_n'][i]:.3f}N")
                if trace["min_volume_ratio"][i]<=.05:
                    text += " | INVALID ELEMENT: not a physical damage model"
                draw.text((10, 5), text, fill="white")
                writer.add_image(np.asarray(frame))
                if i in (0, 79, 159, count-1):
                    frame.save(a.output/f"frame_{i:04d}.png")
    (a.output/"render.json").write_text(json.dumps(dict(trace=str(a.trace.resolve()),
        frames=count, fps=20, camera="palm", mode=a.mode, video=video,
        stop_reason=stop_reason,
        geometry_time_offset_sec=-model.opt.timestep, maximum_vertex_error_m=maximum_vertex_error,
        full_episode_success=success, geometry="true deformable surface; optional massless hand CAD"), indent=2)+"\n")


if __name__ == "__main__":
    main()
