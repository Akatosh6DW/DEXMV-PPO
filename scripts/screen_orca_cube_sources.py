"""Screen source clips against the fixed Orca fixture, without policy selection."""

import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.optimize import least_squares

from diagnose_orca_grasp import model_indices
from prepare_hic_cube_demo import convert
from prepare_hic_orca_object import digest, object_local
from probe_hic53_orca import cube_penetration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_cube_acquisition/source_screen"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    model_path = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    qids, cube_id, tip_ids = model_indices(model)
    with np.load(model_path.parent / "reference.npz") as z:
        packet = {k: z[k] for k in z.files}
    report = dict(model_sha256=digest(model_path), spatial_scale=1.,
                  screening="Fixed object pose/orientation of the current fixture, not a universal reachability verdict",
                  sequences=[])
    for seq in (20, 21):
        source = (ROOT / "artifacts/cube20/hic20/models_motion/20" if seq == 20 else
                  ROOT / "artifacts/orca_cube_acquisition/source21/models/21")
        video_root = ROOT / ("artifacts/cube20/hic20" if seq == 20 else "artifacts/orca_cube_acquisition/source21/videos")
        video = next(video_root.rglob("input_rgb.mp4"))
        video_info = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=r_frame_rate,nb_read_frames,duration",
            "-of", "json", str(video)]))["streams"][0]
        a, b = map(int, video_info["r_frame_rate"].split("/"))
        tracking = convert(source, a/b)
        assert len(tracking["time"]) == int(video_info["nb_read_frames"])
        local = object_local(tracking["hand_joints"], tracking["object_poses"])
        target = local[:, [4, 8]] @ packet["object_to_world_rotation"].T + packet["object_position"]
        selected = np.arange(10, 17)
        lo, hi = model.actuator_ctrlrange[selected].T
        previous = np.clip(packet["initial_ctrl"][selected], lo+.08, hi-.08)
        data = mujoco.MjData(model)
        data.qpos[cube_id:cube_id+7] = np.r_[packet["object_position"], packet["object_quaternion"]]
        errors, controls, frames, depths = [], [], [], []
        for frame in range(0, len(target), 5):
            def residual(q):
                ctrl = packet["initial_ctrl"].copy()
                ctrl[selected] = q
                data.qpos[qids] = ctrl
                mujoco.mj_forward(model, data)
                collision = np.zeros(model.ngeom)
                gid = model.geom("cube").id
                for c in data.contact[:data.ncon]:
                    if gid in (c.geom1, c.geom2):
                        other = c.geom2 if c.geom1 == gid else c.geom1
                        if model.geom_contype[other] == 1:
                            collision[other] = max(collision[other], -c.dist-.0005)
                return np.r_[(data.geom_xpos[tip_ids[:2]]-target[frame]).ravel(),
                             .002*(q-previous), 5*collision]
            fit = least_squares(residual, previous, bounds=(lo+.08, hi-.08), max_nfev=100)
            residual(fit.x)
            errors.append(np.linalg.norm(data.geom_xpos[tip_ids[:2]]-target[frame], axis=1))
            depths.append(cube_penetration(model, data))
            controls.append(data.qpos[qids].copy())
            frames.append(frame)
            previous = fit.x
        errors, depths = np.array(errors), np.array(depths)
        np.savez_compressed(OUT / f"source{seq}_screen.npz", **tracking, object_local=local,
                            sampled_frames=frames, ik_ctrl=controls, errors_m=errors, penetration_m=depths)
        entry = dict(sequence=seq, source=str(source), source_video=str(video), video=video_info,
                     frames=len(tracking["time"]), bone_consistency_max_m=float(tracking["bone_consistency_max_m"]),
                     median_tip_error_mm=(1000*np.median(errors, axis=0)).tolist(),
                     p95_tip_error_mm=(1000*np.quantile(errors,.95,axis=0)).tolist(),
                     reachable_fraction=float(np.mean((errors.max(axis=1)<.015)&(depths<.003))),
                     sha256={f.name:digest(f) for f in source.glob("*.MOTION")}, video_sha256=digest(video))
        report["sequences"].append(entry)
        print(json.dumps(entry), flush=True)
        for frame in (0, len(tracking["time"])//2, len(tracking["time"])-1):
            subprocess.run(["ffmpeg", "-v", "error", "-ss", str(frame/(a/b)), "-i", str(video),
                            "-frames:v", "1", str(OUT / f"source{seq}_frame{frame}.png")], check=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
