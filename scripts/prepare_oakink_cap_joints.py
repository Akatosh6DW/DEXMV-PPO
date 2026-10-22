"""Validate published derived landmarks against the matching official sequence."""

import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from prepare_hic_orca_object import digest, object_local
from prepare_oakink_cap_source import AnnotationReader, SEQUENCE, CAP, BOTTLE
from preview_oakink_cap_source import stream_digest


def load_numeric_npy(payload):
    stream = io.BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError("Unsupported annotation NPY version")
    header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, _, dtype = header(stream)
    if shape != () or dtype != np.dtype(object):
        raise ValueError("Expected an annotation dictionary in a scalar NPY container")
    return AnnotationReader(stream).load().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotation_hash = digest(args.source / "anno_preview.pkl")
    derived_hash = stream_digest(args.source / "derived_annotations.tar")
    if derived_hash != "1da5c7edb3e45c9fb6a920aa781a1c0542ac27606187fdc3bd8d4e73ebba9149":
        raise ValueError("Derived archive does not match the published LFS checksum")
    with (args.source / "anno_preview.pkl").open("rb") as stream:
        source = AnnotationReader(stream).load()
    rows, episode_names, errors = {}, [], []
    with tarfile.open(args.source / "derived_annotations.tar") as archive:
        for entry in archive:
            if SEQUENCE not in entry.name or not entry.name.endswith("right.npy"):
                continue
            d = load_numeric_npy(archive.extractfile(entry).read())
            assert d["source"]["annotation_sha256"] == annotation_hash
            assert d["source"]["sequence_token"] == SEQUENCE and d["anno_type"] == "right"
            assert d["camera"]["extrinsics_convention"] == "world_to_camera"
            episode_names.append(entry.name)
            hand = d["right"]
            for i, frame in enumerate(d["source_frame_id"]):
                frame = int(frame)
                raw = source["raw_mano"][frame]
                q = raw["rh__pose_coeffs"].numpy().reshape(16, 4)
                rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
                errors.append([np.max(abs(rotation[0] - hand["global_orient_worldspace"][i])),
                               np.max(abs(rotation[1:] - hand["hand_pose"][i])),
                               np.max(abs(raw["rh__tsl"].numpy().reshape(3) - hand["transl_worldspace"][i])),
                               np.max(abs(source["cam_extr"]["egocentric"][frame] - d["extrinsics"][i]))])
                np.testing.assert_allclose(raw["rh__betas"].numpy().reshape(10), hand["beta"], atol=1e-6)
                assert hand["annotation_valid"][i] and hand["kept_frames"][i]
                joints = np.asarray(hand["joints_worldspace"][i])
                assert joints.shape == (21, 3) and np.isfinite(joints).all()
                np.testing.assert_allclose(joints[0], hand["transl_worldspace"][i], atol=1e-7)
                transform = source["cam_extr"]["egocentric"][frame]
                camera_joints = joints @ transform[:3, :3].T + transform[:3, 3]
                np.testing.assert_allclose(camera_joints, hand["joints_camspace"][i], atol=2e-6)
                if frame in rows:
                    np.testing.assert_array_equal(rows[frame], joints)
                rows[frame] = joints
    assert np.max(errors) <= 1e-6
    available = sorted(f for f in rows if 239 <= f < 1768)
    expected = [f for f in source["frame_id_list"] if 239 <= f < 1768]
    if available != expected:
        raise ValueError("Missing source frames; do not invent landmarks or silently interpolate gaps")
    frames = np.asarray(available)
    joints = np.stack([rows[f] for f in frames])
    cap = np.stack([source["obj_transf"][CAP][f] for f in frames])
    bottle = np.stack([source["obj_transf"][BOTTLE][f] for f in frames])
    local = object_local(joints, bottle)
    args.output.mkdir(parents=True, exist_ok=False)
    # Use only the original sequence's explicit frame IDs, never the derived
    # clip-local video_decode_frame counter as a source-image index.
    np.savez_compressed(args.output / "human_cap.npz", frame_id=frames, joints_world=joints,
                        joints_bottle=local, cap_world=cap, bottle_world=bottle,
                        cap_in_bottle=np.linalg.inv(bottle) @ cap,
                        source_image_index=np.array([source["frame_id_list"].index(int(f)) for f in frames]),
                        time=(frames - frames[0]) / 120., clock_hz_inferred=120.)
    selected = [min(available, key=lambda f: abs(f - t)) for t in (423, 875, 1311, 1744)]
    sheet = Image.new("RGB", (848 * 2, 480 * 2), "white")
    colors = ["red", "lime", "cyan", "yellow", "magenta"]
    lengths = []
    for finger in range(5):
        chain = [0, *range(finger * 4 + 1, finger * 4 + 5)]
        lengths.append(np.linalg.norm(np.diff(joints[:, chain], axis=1), axis=-1).mean(axis=0).tolist())
    with tarfile.open(args.source / "video_frames.tar") as archive:
        for i, frame in enumerate(selected):
            picture = Image.open(io.BytesIO(archive.extractfile(f"{SEQUENCE}/104422070969/{frame:06d}.png").read())).convert("RGB")
            extrinsic = source["cam_extr"]["egocentric"][frame]
            xyz = rows[frame] @ extrinsic[:3, :3].T + extrinsic[:3, 3]
            uv = xyz @ source["cam_intr"]["egocentric"][frame].T
            uv = uv[:, :2] / uv[:, 2:]
            draw = ImageDraw.Draw(picture)
            for j, point in enumerate(uv):
                x, y = point
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="white")
                draw.text((x + 2, y + 2), str(j), fill="white", stroke_width=1, stroke_fill="black")
            for finger in range(5):
                chain = [0, *range(finger * 4 + 1, finger * 4 + 5)]
                draw.line([tuple(uv[j]) for j in chain], fill=colors[finger], width=2)
            draw.text((10, 10), f"Source frame {frame}: published derived landmarks", fill="white", stroke_width=1, stroke_fill="black")
            picture.save(args.output / f"joints_{frame:06d}.png")
            sheet.paste(picture, ((i % 2) * 848, (i // 2) * 480))
    sheet.save(args.output / "landmark_projection.png")
    report = dict(sequence=SEQUENCE, frames=len(frames), source_frame_start=int(frames[0]), source_frame_end=int(frames[-1]),
                  full_selected_image_coverage=True, unique_derived_episodes=len(episode_names),
                  max_pose_translation_camera_errors=np.max(errors, axis=0).tolist(),
                  mean_assumed_chain_bone_lengths_m=lengths, landmark_order_requires_visual_check=True,
                  original_annotation_sha256=annotation_hash, derived_archive_sha256=derived_hash,
                  packet_sha256=digest(args.output / "human_cap.npz"),
                  timing="30 fps images; 120 Hz frame-ID clock inferred in derived source, not a measured sensor clock",
                  derived_source="https://huggingface.co/datasets/MIT-Media-Lab/oakink2-annotations-v1",
                  derived_license="CC-BY-NC-4.0; retain source credit; research use only under published terms",
                  limits="Joint coordinates were published by the derived dataset, not recomputed from MANO here. Cross-checks validate shared parameters and transforms, not absolute hand-fit accuracy.",
                  official_source="https://huggingface.co/datasets/kelvin34501/OakInk-v2")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
