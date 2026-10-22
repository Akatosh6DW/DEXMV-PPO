"""Export the compiled benchmark physics, reference and dataset schemas."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    model_path = ROOT / "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
    reference = ROOT / "artifacts/orca_cube_acquisition/E01_flex50_refinement/candidate_reference.npz"
    dataset = ROOT / "artifacts/orca_cube_acquisition/D01_quality_demos/trajectories.npz"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    actuators = []
    for i in range(model.nu):
        a = model.actuator(i)
        joint = model.joint(int(model.actuator_trnid[i, 0]))
        assert int(joint.type[0]) == mujoco.mjtJoint.mjJNT_HINGE
        np.testing.assert_array_equal(a.gear, [1, 0, 0, 0, 0, 0])
        actuators.append(dict(index=i, name=a.name, joint=joint.name,
                              joint_range_rad=joint.range.tolist(), ctrl_range_rad=a.ctrlrange.tolist(),
                              force_range=a.forcerange.tolist(), force_limited=bool(a.forcelimited[0]),
                              gear=a.gear.tolist(), gain_parameters=a.gainprm.tolist(),
                              bias_parameters=a.biasprm.tolist(), dynamics_parameters=a.dynprm.tolist(),
                              joint_damping=model.dof_damping[int(joint.dofadr[0])].item(),
                              joint_armature=model.dof_armature[int(joint.dofadr[0])].item()))
    geoms = []
    for i in range(model.ngeom):
        geom = model.geom(i)
        if not geom.contype[0] and not geom.conaffinity[0]:
            continue
        geoms.append(dict(index=i, name=geom.name, body=model.body(int(geom.bodyid[0])).name,
                          type=int(geom.type[0]), size=geom.size.tolist(), position=geom.pos.tolist(),
                          quaternion_wxyz=geom.quat.tolist(), friction=geom.friction.tolist(),
                          contype=int(geom.contype[0]), conaffinity=int(geom.conaffinity[0]),
                          condim=int(geom.condim[0]), margin=float(geom.margin[0]), gap=float(geom.gap[0]),
                          solref=geom.solref.tolist(), solimp=geom.solimp.tolist()))
    with np.load(reference) as z:
        reference_card = dict(schema={k: dict(shape=z[k].shape, dtype=str(z[k].dtype)) for k in z.files},
                              initial_ctrl=z["initial_ctrl"].tolist(),
                              object_position_m=z["object_position"].tolist(),
                              object_quaternion_wxyz=z["object_quaternion"].tolist(),
                              object_extents_m=z["object_extents_m"].tolist(),
                              refinement_delta_rad=z["refinement_delta"].tolist(),
                              first_time_sec=float(z["time"][0]), last_time_sec=float(z["time"][-1]))
    with np.load(dataset) as z:
        data_card = dict(schema={k: dict(shape=z[k].shape, dtype=str(z[k].dtype)) for k in z.files},
                         active_samples=int(z["active"].sum()), episodes=len(z["active"]))
    cube = model.body("cube")
    result = dict(version="S02", nq=model.nq, nv=model.nv, nu=model.nu,
                  model=str(model_path.relative_to(ROOT)), model_sha256=digest(model_path),
                  gravity_m_per_s2=model.opt.gravity.tolist(), physics_dt_sec=model.opt.timestep,
                  integrator=int(model.opt.integrator), solver=int(model.opt.solver),
                  solver_iterations=model.opt.iterations, solver_tolerance=model.opt.tolerance,
                  cone=int(model.opt.cone), cube_mass_kg=float(cube.mass[0]), cube_inertia=cube.inertia.tolist(),
                  actuators=actuators, contact_geometries=geoms,
                  reference=reference_card, reference_sha256=digest(reference), dataset=data_card,
                  dataset_sha256=digest(dataset), protocol_sha256=digest(root / "protocol.json"),
                  benchmark_reset_parameters=json.loads((root / "reset_geometry_audit.json").read_text()),
                  units="MuJoCo SI units; angles radians; friction = sliding, torsional, rolling; quaternions wxyz",
                  actuator_force_units="N*m for these unit-gear hinge transmissions, not fingertip contact force N",
                  note="Base compiled XML plus reference metadata. Runtime initial hand/object pose is reset from reference, not XML qpos0. Reward config inheritance is described in docs/orca_benchmark_data_guide.md.")
    (root / "benchmark_card.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# S02 实际模型参数", "", f"nq={model.nq}, nv={model.nv}, nu={model.nu}。",
             "nq包含自由物体四元数，nv包含其6维速度；nu才是动作执行器数量。",
             f"方块质量：{float(cube.mass[0])*1000:.1f} g；物理步长：{model.opt.timestep*1000:.1f} ms。",
             "完整摩擦、求解器、碰撞体和数据形状见同目录benchmark_card.json。", "",
             "| 动作索引 | 执行器 | 关节 | 控制范围rad | 初始目标rad | 执行器限幅N*m |", "|---|---|---|---|---|---|"]
    for a in actuators:
        i = a["index"]
        limits = a["force_range"] if a["force_limited"] else "未启用"
        lines.append(f"| {i} | {a['name']} | {a['joint']} | {a['ctrl_range_rad']} | "
                     f"{reference_card['initial_ctrl'][i]:.6f} | {limits} |")
    lines += ["", "## 数据规模", "", f"示范回合：{data_card['episodes']}，有效状态动作样本：{data_card['active_samples']}。",
              "动作顺序以上表为准，不能按手指名称自行重新排序后直接输入网络。",
              "参考文件time长度包含40秒末姿态保持，不等于40秒不同的视频动作。",
              "固定的是腕座安装位置；动作0仍可驱动一个腕转动关节，不是17个关节都属于手指。",
              "名称中的p/r/m/i/t分别指小指、无名指、中指、食指、拇指；abd为外展，mcp为掌指关节，pip为近端指间关节，cmc为拇指腕掌关节。",
              "本模型是单位gear的旋转关节传动，因此执行器限幅为力矩N*m，不能与指尖接触力N直接比较。",
              "名义力限幅不等于指尖接触力上限，接触力由姿态、刚度和接触求解共同决定。"]
    (root / "benchmark_card.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(nq=model.nq, nv=model.nv, nu=model.nu, contacts=len(geoms), dataset=data_card)))


if __name__ == "__main__":
    main()
