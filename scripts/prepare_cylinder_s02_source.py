"""Decode official DexMV can trajectories without changing the RL pipeline."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np

MEMBER = "demonstrations/relocate-tomato_soup_can.pkl"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArrayUnpickler(pickle.Unpickler):
    """This published file needs only NumPy's numeric-array constructors."""

    def find_class(self, module, name):
        allowed = {("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
                   ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unexpected global: {module}.{name}")
        return allowed[module, name]


def load_archive(path):
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(MEMBER)
    result = ArrayUnpickler(io.BytesIO(raw)).load()
    if not isinstance(result, dict) or not result:
        raise ValueError("Expected named demonstration sequences")
    return result, hashlib.sha256(raw).hexdigest()


def compatible_xml(path):
    root = ET.parse(path).getroot()
    # Removed MuJoCo 2 metadata; no kinematic parameter is changed.
    root.find("option").attrib.pop("apirate", None)
    root.find("size").attrib.pop("nstack", None)
    for camera in root.findall(".//camera[@quat]"):
        camera.set("quat", " ".join(camera.get("quat").replace(",", " ").split()))
    return ET.tostring(root, encoding="unicode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    demonstrations, member_hash = load_archive(args.archive)
    xml = compatible_xml(args.model)
    model = mujoco.MjModel.from_xml_string(xml)
    if (model.nq, model.nv, model.nbody, model.nsite) != (37, 36, 29, 28):
        raise ValueError("Official archive and model layout do not match")
    obj = model.body("tomato_soup_can_0").id
    sites = [model.site("S_"+prefix+"tip").id for prefix in ("th", "ff", "mf", "rf", "lf")]
    palm = model.body("palm").id
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"source_model.xml").write_text(xml)
    records = []
    names = sorted(demonstrations, key=lambda name: int(name.split("/")[0].removeprefix("seq_")))
    for name in names:
        demo = demonstrations[name]
        qpos = np.stack([state["qpos"] for state in demo["sim_data"]])
        if qpos.shape != (len(demo["observations"]), 37) or not np.isfinite(qpos).all():
            raise ValueError("Invalid sequence " + name)
        np.testing.assert_allclose(np.linalg.norm(qpos[:, 33:37], axis=1), 1., atol=1e-6)
        for field in ("body_pos", "body_quat", "site_pos", "site_quat"):
            value = demo["model_data"][0][field]
            if value.shape != getattr(model, field).shape or not np.isfinite(value).all():
                raise ValueError("Invalid per-sequence model data")
            getattr(model, field)[:] = value
        data = mujoco.MjData(model)
        local, rotations, positions, palm_local, palm_rotation = [], [], [], [], []
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            rotation = data.xmat[obj].reshape(3, 3).copy()
            position = data.xpos[obj].copy()
            local.append((data.site_xpos[sites]-position) @ rotation)
            palm_local.append((data.xpos[palm]-position) @ rotation)
            palm_rotation.append(rotation.T @ data.xmat[palm].reshape(3, 3))
            rotations.append(rotation)
            positions.append(position)
        local = np.asarray(local)
        sequence = int(name.split("/")[0].removeprefix("seq_"))
        path = args.output/f"seq_{sequence:03d}.npz"
        np.savez_compressed(path, qpos=qpos, fingertips_object_local=local,
                            object_rotation=rotations, object_position=positions,
                            palm_object_local=palm_local, palm_rotation_object_local=palm_rotation,
                            time=np.arange(len(qpos))*.01, dt=.01,
                            source_sequence=np.array(name))
        records.append(dict(sequence=sequence, source_key=name, frames=len(qpos),
                            duration_sec=(len(qpos)-1)*.01, packet=path.name, sha256=digest(path),
                            thumb_index_distance_min_m=float(np.linalg.norm(local[:, 0]-local[:, 1], axis=1).min()),
                            max_tip_displacement_per_frame_m=float(np.linalg.norm(np.diff(local, axis=0), axis=-1).max())))
    manifest = dict(source_url="https://drive.google.com/file/d/1v-SezFDQBcgekHZBlqulqa8rIgn0iwRT/view",
                    official_repository="https://github.com/yzqin/dexmv-sim", archive_sha256=digest(args.archive),
                    member=MEMBER, member_sha256=member_hash, exported_xml_sha256=digest(args.model),
                    compatible_xml_sha256=digest(args.output/"source_model.xml"),
                    source_type="Official video-derived Adroit demonstrations; not raw RGB inference or PPO weights",
                    time_source="DexMV YCBRelocate control_freq=100; processed/interpolated frames, not camera FPS",
                    fingers=["thumb", "index", "middle", "ring", "little"],
                    coordinates="Same-frame fingertip site centers in the moving object's body frame",
                    selection="All 101 source sequences decoded before any Orca simulation; no reward-based selection",
                    source_actions_used=False, records=records)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(sequences=len(records), frames=sum(r["frames"] for r in records),
                         output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
