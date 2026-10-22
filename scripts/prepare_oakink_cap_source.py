"""Inspect a real, paired bottle/cap sequence without inventing hand landmarks."""

import argparse
from collections import OrderedDict
import io
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
import torch
import trimesh

from prepare_hic_orca_object import digest

SEQUENCE = "scene_01__A008++seq__709821691ed6e9de0ecf__2023-04-23-11-38-59"
CAP = "O02@0015@00016"
BOTTLE = "O02@0015@00017"


class AnnotationReader(pickle.Unpickler):
    """Only the numeric containers present in the published preview format."""
    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
            return np._core.multiarray._reconstruct
        if (module, name) in (("numpy", "ndarray"), ("numpy", "dtype")):
            return getattr(np, name)
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return torch._utils._rebuild_tensor_v2
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return lambda data: torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
        raise pickle.UnpicklingError(f"Unsupported annotation constructor: {module}.{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.source / "anno_preview.pkl").open("rb") as f:
        annotation = AnnotationReader(f).load()
    with tarfile.open(args.source / "program.tar") as archive:
        descriptions = json.load(archive.extractfile(f"program/desc_info/{SEQUENCE}.json"))
        programs = json.load(archive.extractfile(f"program/program_info/{SEQUENCE}.json"))
    key = "((435, 1612), (423, 1768))"
    assert programs[key]["primitive"] == "unscrew"
    assert programs[key]["interaction_mode"] == "rh_main"
    assert set(programs[key]["obj_list"]) == {CAP, BOTTLE}
    frames = [k for k in annotation["mocap_frame_id_list"] if 423 <= k < 1768]
    assert frames and all(k in annotation["obj_transf"][CAP] and k in annotation["raw_mano"] for k in frames)
    transforms = {obj: np.stack([annotation["obj_transf"][obj][k] for k in frames]) for obj in (CAP, BOTTLE)}
    relative = np.linalg.inv(transforms[BOTTLE]) @ transforms[CAP]
    np.testing.assert_allclose(relative[:, 3], np.tile([0., 0., 0., 1.], (len(frames), 1)), atol=1e-6)
    hand = {name: np.stack([annotation["raw_mano"][k][name].detach().cpu().numpy().squeeze(0)
                           for k in frames]) for name in ("rh__pose_coeffs", "rh__tsl", "rh__betas")}
    args.output.mkdir(parents=True, exist_ok=False)
    meshes = {}
    with tarfile.open(args.source / "object_raw.tar") as archive:
        descriptions_obj = json.load(archive.extractfile("object_raw/obj_desc.json"))
        for obj, label in ((CAP, "cap"), (BOTTLE, "bottle")):
            data = archive.extractfile(f"object_raw/align_ds/{obj}/scan.ply").read()
            path = args.output / (label + ".ply")
            path.write_bytes(data)
            mesh = trimesh.load(io.BytesIO(data), file_type="ply", force="mesh", process=False)
            meshes[label] = dict(id=obj, description=descriptions_obj[obj], extents_native=mesh.extents.tolist(),
                                 vertices=len(mesh.vertices), faces=len(mesh.faces),
                                 watertight=bool(mesh.is_watertight), sha256=digest(path))
    np.savez_compressed(args.output / "source_motion.npz", mocap_frame_id=np.asarray(frames),
                        cap_world=transforms[CAP], bottle_world=transforms[BOTTLE],
                        cap_in_bottle=relative, **hand)
    report = dict(sequence=SEQUENCE, selection="Smallest published image archive among bottle-unscrewing sequences; no policy result used",
                  segment=programs[key], description=descriptions[key], mocap_interval=[423, 1768],
                  captured_frames=len(annotation["frame_id_list"]), mocap_frames=len(frames),
                  camera_definition=annotation["cam_def"], camera_selection=annotation["cam_selection"], meshes=meshes,
                  source_sha256={p.name: digest(p) for p in args.source.iterdir()
                                 if p.is_file() and p.name != "video_frames.tar"},
                  motion_sha256=digest(args.output / "source_motion.npz"),
                  video_archive_verified=False,
                  provenance="OakInk2 published synchronized images and mocap-based hand/object fits; not an RGB estimator rerun",
                  hand_landmarks_ready=False, required="Licensed MANO_RIGHT.pkl v1.2 to instantiate the supplied right-hand pose/shape",
                  license="Dataset card: CC-BY-SA-4.0; MANO model has separate terms",
                  official_toolkit="https://github.com/oakink/OakInk2",
                  official_dataset="https://huggingface.co/datasets/kelvin34501/OakInk-v2")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
