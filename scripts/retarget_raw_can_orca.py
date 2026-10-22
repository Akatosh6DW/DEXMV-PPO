"""Fresh left-human/right-Orca landmark IK, with no dynamics/reward search."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
BASES = ("right_wrist", "right_i-mcp", "right_m-mcp", "right_p-mcp")
HUMAN_BASES = [0, 5, 9, 17]
HUMAN_TIPS = [4, 8, 12, 16, 20]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def palm_basis(bases):
    wrist, index, middle, pinky = bases
    x = index-pinky
    x /= np.linalg.norm(x)
    y = middle-wrist
    y -= x*np.dot(x, y)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = dict(np.load(args.packet, allow_pickle=False))
    model = mujoco.MjModel.from_xml_path(str(args.hand_model))
    data = mujoco.MjData(model)
    qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    base_ids = [model.joint(n).id for n in BASES]
    tip_ids = [model.geom(f"orca_{n}_tip_ball_proxy").id for n in FINGERS]
    low, high = model.actuator_ctrlrange.T
    initial = np.clip(np.zeros(model.nu), low, high)
    data.qpos[qids] = initial
    mujoco.mj_kinematics(model, data)
    robot_bases = data.xanchor[base_ids].copy()
    robot_frame = palm_basis(robot_bases)
    selected = source["selected_frames"]
    frames = source["hand_joints"]
    if set(source["handedness"][selected]) != {"Left"}:
        raise ValueError("This explicit mirror pipeline expects a left-hand source")
    # Reflect palm-normal coordinate. Proper rotations alone cannot change chirality.
    mirror = np.diag([1., 1., -1.])
    def transform(i, points):
        bases = frames[i, HUMAN_BASES]
        matrix = robot_frame @ mirror @ palm_basis(bases).T
        return (points-bases.mean(0)) @ matrix.T + robot_bases.mean(0), matrix
    targets = np.array([transform(i, frames[i, HUMAN_TIPS])[0] for i in selected])
    controls, fitted, errors = [], [], []
    previous = initial.copy()
    for target in targets:
        def residual(ctrl):
            data.qpos[qids] = np.r_[0., ctrl]
            mujoco.mj_kinematics(model, data)
            return np.r_[(data.geom_xpos[tip_ids]-target).ravel(), .002*(ctrl-previous[1:])]
        fit = least_squares(residual, np.clip(previous[1:], low[1:]+1e-7, high[1:]-1e-7),
                            bounds=(low[1:], high[1:]), max_nfev=150)
        residual(fit.x)
        previous = np.r_[0., fit.x]
        controls.append(previous.copy())
        fitted.append(data.geom_xpos[tip_ids].copy())
        errors.append(np.linalg.norm(fitted[-1]-target, axis=1))
    controls, fitted, errors = map(np.asarray, (controls, fitted, errors))
    anchor = int(source["anchor_frame"])
    position, matrix = transform(anchor, source["object_centers"][anchor])
    axis = matrix @ source["object_axes"][anchor]
    axis /= np.linalg.norm(axis)
    rotation, _ = Rotation.align_vectors(axis[None], np.array([[0., 0., 1.]]))
    quaternion = rotation.as_quat()[[3, 0, 1, 2]]
    radius, half = float(source["diameter"])/2, float(source["height"])/2
    extent = half*abs(axis[2]) + radius*np.sqrt(1-axis[2]**2)
    root = ET.parse(args.hand_model).getroot()
    root.set("model", "Orca raw-video tomato can grasp")
    root.find("option").set("cone", "elliptic")
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    for asset in root.findall("asset/mesh"):
        name = Path(asset.get("file"))
        if not name.is_absolute():
            asset.set("file", str((args.hand_model.parent/meshdir/name).resolve()))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    body = root.find(".//body[@name='cube']")
    body.set("pos", " ".join(map(str, position)))
    body.set("quat", " ".join(map(str, quaternion)))
    geom = body.find("geom[@name='cube']")
    geom.attrib.pop("mesh", None)
    geom.set("type", "cylinder")
    geom.set("size", f"{radius} {half}")
    geom.set("mass", ".035")  # Task mass assumption, NOT inferred from RGB.
    support = position-[0, 0, extent+.004]
    root.find(".//body[@name='cube_support']").set("pos", " ".join(map(str, support)))
    model_path = args.output/"model.xml"
    ET.ElementTree(root).write(model_path, encoding="unicode")
    compiled = mujoco.MjModel.from_xml_path(str(model_path))
    test = mujoco.MjData(compiled)
    test.qpos[qids] = initial
    mujoco.mj_forward(compiled, test)
    cube = compiled.geom("cube").id
    penetration = max([0.]+[-float(c.dist) for c in test.contact[:test.ncon]
                           if cube in (c.geom1, c.geom2) and
                           compiled.geom_contype[c.geom2 if c.geom1 == cube else c.geom1] == 1])
    reference = args.output/"reference.npz"
    np.savez_compressed(reference, ctrl=controls, time=source["time"][selected]-source["time"][selected[0]],
                        source_frames=selected, target_tips=targets, fitted_tips=fitted, errors_m=errors,
                        object_position=position, object_quaternion=quaternion,
                        object_extents_m=[2*radius, 2*radius, 2*half], initial_ctrl=initial,
                        spatial_scale=1., object_to_world_rotation=rotation.as_matrix())
    report = dict(source_packet_sha256=digest(args.packet), hand_model_sha256=digest(args.hand_model),
                  script_sha256=digest(__file__), model_sha256=digest(model_path),
                  reference_sha256=digest(reference), source_frames=selected.tolist(),
                  mapping="Left-to-right mirror plus per-frame palm alignment and bounded fingertip IK",
                  source_object="DexMV raw RGB tomato soup can 41", target_object="unscaled analytic cylinder",
                  dimensions_m=[2*radius, 2*radius, 2*half], assumed_mass_kg=.035,
                  mean_tip_error_mm=(errors.mean(0)*1000).tolist(),
                  p95_tip_error_mm=(np.quantile(errors, .95, axis=0)*1000).tolist(),
                  initial_hand_object_penetration_m=penetration,
                  object_position=position.tolist(), object_axis=axis.tolist(),
                  no_reward_search=True, no_old_reference=True,
                  limitations=["Direct Orca adaptation of pose-to-robot pipeline, not original Adroit/DAPG policy",
                               "Human arm motion removed; source segment is 2 to 4 seconds",
                               "Final reference held after its two-second motion; video does not contain 40 seconds of holding",
                               "All methods reset to neutral zero joint targets, not a video pregrasp",
                               "Fingertip markers approximate pad locations; IK accuracy is not contact validity"])
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
