"""Inspect synchronized source images with projected source-object outlines."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from prepare_oakink_cap_source import AnnotationReader, SEQUENCE, CAP, BOTTLE


def stream_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.source / "anno_preview.pkl").open("rb") as stream:
        data = AnnotationReader(stream).load()
    video_hash = stream_digest(args.source / "video_frames.tar")
    expected_hash = "23909785f366fa9c1c1b249c0d7e67b3435c9888efbda2845741d15d73473391"
    assert video_hash == expected_hash, "Source video archive differs from the official LFS checksum"
    args.output.mkdir(parents=True, exist_ok=False)
    frames = np.array(data["frame_id_list"])
    selected = [int(frames[np.argmin(abs(frames - k))]) for k in (423, 875, 1311, 1767)]
    mesh = {CAP: trimesh.load(args.prepared / "cap.ply", process=False).vertices,
            BOTTLE: trimesh.load(args.prepared / "bottle.ply", process=False).vertices}
    cameras = data["cam_def"]
    sheet = Image.new("RGB", (4 * 480, 4 * 294), "white")
    projected = []
    with tarfile.open(args.source / "video_frames.tar") as archive:
        for row, (serial, camera) in enumerate(cameras.items()):
            for col, frame in enumerate(selected):
                path = f"{SEQUENCE}/{serial}/{frame:06d}.png"
                picture = Image.open(io.BytesIO(archive.extractfile(path).read())).convert("RGB")
                picture.save(args.output / f"{camera}_{frame:06d}_original.png")
                draw = ImageDraw.Draw(picture)
                for obj, color in ((CAP, "red"), (BOTTLE, "cyan")):
                    transform = data["cam_extr"][camera][frame] @ data["obj_transf"][obj][frame]
                    points = mesh[obj] @ transform[:3, :3].T + transform[:3, 3]
                    pixels = points @ data["cam_intr"][camera][frame].T
                    pixels = pixels[:, :2] / pixels[:, 2:3]
                    box = [*pixels.min(axis=0), *pixels.max(axis=0)]
                    if np.all(points[:, 2] > 0):
                        draw.rectangle(box, outline=color, width=3)
                    projected.append(dict(camera=camera, frame=frame, object=obj, box=box,
                                          all_in_front=bool(np.all(points[:, 2] > 0))))
                picture.thumbnail((480, 270))
                tile = Image.new("RGB", (480, 294), "white")
                tile.paste(picture, (0, 24))
                ImageDraw.Draw(tile).text((5, 5), f"{camera} frame {frame}: red cap / cyan bottle", fill="black")
                sheet.paste(tile, (480 * col, 294 * row))
    sheet.save(args.output / "source_object_projection.png")
    report = dict(sequence=SEQUENCE, frames=selected, video_sha256=video_hash,
                  source_archive_lfs_hash_verified=True, projection=projected,
                  source_images="Original images retained; overlaid boxes use source mesh and source camera/object transforms",
                  note="Projected object boxes verify correspondence only; hand landmarks and policy not yet validated")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "source_object_projection.png")


if __name__ == "__main__":
    main()
