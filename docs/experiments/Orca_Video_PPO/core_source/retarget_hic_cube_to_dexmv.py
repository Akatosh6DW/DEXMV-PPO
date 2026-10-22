#!/usr/bin/env python3
# 中文阅读副本：01 视频跟踪到DexMV中间手型
# 原始入口：scripts/retarget_hic_cube_to_dexmv.py
# 仅用于阅读，保留原路径假设；不要从此副本启动训练。运行请使用原始入口。
# 本副本只增加注释，生成时核验AST与非注释词法序列完全一致。
"""Retarget HIC cube motion with DexMV's constrained position optimizer."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from hand_imitation.env.models.base import MujocoXML
from hand_imitation.env.utils.mjcf_utils import xml_path_completion
from hand_imitation.kinematics.optimizer import PositionOptimizer


# 导读：建立DexMV的受关节限制位置优化器；这里做运动学拟合，不是TRPO或PPO训练。
def build_optimizer():
    model = MujocoXML(xml_path_completion("adroit/adroit_relocate.xml"))
    names, indices = ["palm"], [0]
    for f, prefix in enumerate(["th", "ff", "mf", "rf", "lf"]):
        names.extend([prefix + "proximal", prefix + "middle", "hic_" + prefix + "tip"])
        indices.extend([1 + 4 * f, 2 + 4 * f, 4 + 4 * f])
        parent = model.worldbody.find(".//body[@name='%sdistal']" % prefix)
        site = parent.find("site[@name='S_%stip']" % prefix)
        ET.SubElement(parent, "body", name="hic_" + prefix + "tip", pos=site.get("pos"))
    # The moving hand needs a larger root workspace than the table-relocate task.
    # Finger limits remain unchanged; training must use the same exported XML.
    joints = model.worldbody.findall(".//joint")
    for j in joints[:6]:
        limit = "-1 1" if j.get("type") == "slide" else "-3.14 3.14"
        j.set("range", limit)
        actuator = model.actuator.find("*[@joint='%s']" % j.get("name"))
        if actuator is not None:
            actuator.set("ctrlrange", limit)
    full_xml = model.get_xml()
    for section in [model.asset, model.sensor, model.tendon, model.equality, model.contact, model.actuator]:
        model.root.remove(section)
    for parent in model.worldbody.iter():
        for child in list(parent):
            if child.tag in ["geom", "site"]:
                parent.remove(child)
    opt = PositionOptimizer(model.get_xml(), names)
    limits_by_name = {j.get("name"): np.fromstring(j.get("range"), sep=" ") for j in joints}
    limits = np.array([limits_by_name[j.name] for j in opt.chain.get_variable_joints()])
    opt.set_joint_limit(limits)
    opt.opt.set_maxeval(150)
    return opt, limits, np.array(indices), full_xml


def rigid_alignment(source, target):
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    fix = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
    r = vt.T @ fix @ u.T
    return r, b - r @ a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("artifacts/cube20/hic20/prepared/human_cube.npz"))
    p.add_argument("--output", type=Path, default=Path("artifacts/cube20/hic20/dexmv_retarget"))
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--contact-weight", type=int, default=0)
    a = p.parse_args()
    source = np.load(a.input)
    joints = source["hand_joints"].astype(float)
    objects = source["object_poses"].astype(float)
    if a.frames:
        joints, objects = joints[:a.frames], objects[:a.frames]
    opt, limits, indices, full_xml = build_optimizer()
    q = np.clip(np.zeros(opt.dof), limits[:, 0], limits[:, 1])
    poses = opt.chain.forward_kinematics(q)
    anchors = ["palm", "ffproximal", "mfproximal", "rfproximal", "lfproximal"]
    r, t = rigid_alignment(joints[0, [0, 5, 9, 13, 17]], np.array([poses[n][:3, 3] for n in anchors]))
    target = (joints @ r.T + t)[:, indices]
    world_transform = np.eye(4)
    world_transform[:3, :3], world_transform[:3, 3] = r, t
    # 导读：手和物体必须使用同一个坐标变换，否则指尖与物体会错位。
    objects = world_transform @ objects
    optimization_target = target
    if a.contact_weight:
        tips = target[:, [3, 6]]
        local = np.einsum("tji,tkj->tki", objects[:, :3, :3], tips - objects[:, None, :3, 3])
        surface = np.clip(local, -.0265, .0265)
        gap = local - surface
        distance = np.linalg.norm(gap, axis=-1, keepdims=True)
        desired = surface + gap / np.maximum(distance, 1e-9) * .001
        # Only refine near-contact phases; retain the demonstrated open-hand approach.
        alpha = np.clip((.035 - distance.max(axis=1, keepdims=True)) / .020, 0, 1)
        local = local + alpha * (desired - local)
        refined = np.einsum("tij,tkj->tki", objects[:, :3, :3], local) + objects[:, None, :3, 3]
        optimization_target = np.concatenate([target] + [refined] * a.contact_weight, axis=1)
        opt.body_names += ["hic_thtip", "hic_fftip"] * a.contact_weight
    results, errors, fitted = [], [], []
    # 导读：上一帧关节解作为下一帧起点，逐帧求可实现的动作；拟合误差仍需另外检查。
    for i, desired in enumerate(optimization_target):
        q = opt.retarget(desired, q, verbose=False)
        actual = opt.chain.forward_kinematics(q)
        actual = np.array([actual[n][:3, 3] for n in opt.body_names[:16]])
        results.append(q.copy())
        fitted.append(actual)
        errors.append(np.linalg.norm(actual - target[i], axis=1))
        if i % 30 == 0:
            print("frame %d mean_error %.5f m" % (i, np.mean(errors[-1])), flush=True)
    results, errors = np.array(results), np.array(errors)
    if not np.isfinite(results).all():
        raise ValueError("Nonfinite retarget output")
    violation = np.maximum(limits[:, 0] - results, results - limits[:, 1]).clip(0).max()
    a.output.mkdir(parents=True, exist_ok=True)
    # 导读：文件保存关节、拟合点、目标点和物体姿态；后续Orca重定向用空间关系，不直接复制关节角。
    np.savez_compressed(a.output / "adroit_cube.npz", adroit_qpos=results,
                        object_poses=objects, hand_joints=joints @ r.T + t,
                        target_positions=target, fitted_positions=np.array(fitted),
                        errors_m=errors, joint_limits=limits, camera_to_adroit=world_transform,
                        dt=source["dt"], time=source["time"][:len(results)])
    (a.output / "adroit_hic.xml").write_text(full_xml)
    meta = {"input": str(a.input.resolve()), "frames": len(results),
            "method": "DexMV PositionOptimizer; palm/MCP/PIP/tip positions; fixed rigid alignment",
            "root_limits_expanded": True, "finger_limits_changed": False,
            "contact_weight": a.contact_weight,
            "mean_error_m": float(errors.mean()), "max_error_m": float(errors.max()),
            "tip_mean_error_m": float(errors[:, [3, 6, 9, 12, 15]].mean()),
            "joint_limit_violation": float(violation),
            "initial_cube_minus_palm_m": (objects[0, :3, 3] - fitted[0][0]).tolist()}
    (a.output / "metrics.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
