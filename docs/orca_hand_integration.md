# Orca Hand Integration

## 资产位置

- PPO 侧主位置：`Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/orca_hand/`
- Orca 原始模型：`.../orca_hand/xmls/orcahand_description/`
- DexMV 侧入口：`DexMV/hand_imitation/env/models/assets/orca_hand/orcahand_description`

DexMV 侧使用符号链接指向 PPO 侧同一份 Orca 模型，避免重复存储约 205 MB mesh。

## 模型区分

- `v2/scene_right.xml`：Orca v2 右手完整可视化/碰撞模型。
- `v2/scene_left.xml`：Orca v2 左手完整可视化/碰撞模型。
- `v2/scene_combined.xml`：Orca v2 双手完整模型。
- `v2/scene_right_mjx_pose.xml`：PPO smoke 用 MJX 姿态跟踪模型。

`scene_right_mjx_pose.xml` 保留 17 个关节和 17 个 position actuators，但丢弃 mesh geoms。原因是原始 mesh 直接进 MJX 时编译和训练非常慢；后续如果要做真实抓取，需要再做简化碰撞体、凸分解或 capsule 近似。

## PPO 环境

已注册两个环境：

- `OrcaHandPoseTrack`
- `OrcaHandDexMVResidualPoseTrack`

`OrcaHandPoseTrack` 是纯 Orca 关节姿态跟踪环境。`OrcaHandDexMVResidualPoseTrack` 使用 DexMV 视频提取出的 Orca 17 维控制先验，并让 PPO 输出 residual 修正量：

```text
final_ctrl[t] = orca_prior_ctrl[t] + PPO_action[t] * action_scale
```

当前任务是“接通手型和 DexMV 先验”的基础层，不是最终 cube 抓取环境。最终 cube 抓取还需要 Orca 专用碰撞几何、物体放置、接触奖励和成功指标。

## DexMV 到 Orca 先验

生成脚本：

```bash
python scripts/retarget_dexmv_video_to_orca_prior.py
```

默认输入：

```text
DexMV/examples/retargeting_source/relocate_mustard_example_seq/hand_pose
```

默认输出：

```text
artifacts/ppo_bridge/demo_dataset_orca_direct.npz
```

关键字段：

- `orca_prior_ctrl`: `[T, 17]`，Orca actuator 控制序列。
- `orca_prior_action`: `[T, 17]`，相对默认控制的归一化动作。
- `dexmv_hand_joints`: `[T, 21, 3]`，DexMV 视频手部关节点。
- `orca_ctrl_names`: 17 个 Orca 控制通道名称。

当前生成结果：`orca_prior_ctrl` 为 `122 x 17`，`dt=0.05`。

## Smoke Test

完整 smoke 命令：

```bash
./scripts/run_dexmv_to_orca_residual_ppo_smoke.sh
```

已跑通结果：

- env: `OrcaHandDexMVResidualPoseTrack`
- log: `Aero-Hand/sim_rl/mujoco_playground/logs/OrcaHandDexMVResidualPoseTrack-20260519-012722-orca_dexmv_residual_smoke`
- PPO smoke reward: step `0` 为 `34.096`，step `2560` 为 `33.345`
- DexMV asset smoke: `xml_path_completion('orca_hand/orcahand_description/v2/scene_right.xml')` 能解析到 Orca v2 右手 XML。

这个结果只证明 PPO 训练入口可用，不代表 Orca 已经能抓 cube。

DexMV 的 `mujoco_py` 环境在本机需要：

```bash
export LD_LIBRARY_PATH=/home/ll/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
```

## Feasibility Validation: 2026-05-19

本轮验证目标是确认 Orca 手已经接入 PPO/DexMV/展示链路，而不是验证最终 cube 抓取性能。

执行内容：

- 重新生成 DexMV -> Orca 先验：`artifacts/ppo_bridge/demo_dataset_orca_direct.npz`
- 验证 DexMV 资产入口：`orca_hand/orcahand_description/v2/scene_right.xml`
- 验证 PPO 环境：`OrcaHandDexMVResidualPoseTrack` 可 `reset/step`
- 执行一次短 PPO smoke：`./scripts/run_dexmv_to_orca_residual_ppo_smoke.sh`
- 用 Aero-Hand 展示脚本导出 Orca 模型预览图

验证结果：

- `demo_dataset_orca_direct.npz`: `orca_prior_ctrl` shape 为 `122 x 17`
- DexMV XML 解析：通过，模型名为 `ORCA hand right scene`
- PPO 环境 step：通过，`action_size=17`，`obs['state']=(87,)`，`obs['privileged_state']=(138,)`
- PPO smoke log：`Aero-Hand/sim_rl/mujoco_playground/logs/OrcaHandDexMVResidualPoseTrack-20260519-013415-orca_dexmv_residual_smoke`
- PPO smoke metrics：
  - step `0`: reward `34.0958`，episode cumulative pose_error `5.3917`
  - step `2560`: reward `33.3446`，episode cumulative pose_error `6.0111`
  - avg_episode_length: `80.0`

结论：

- 当前 Orca 手模型、DexMV 先验文件、PPO residual pose-track 环境、训练脚本入口已经连通。
- smoke 指标有限，且 pose-track 不是 cube grasp；这只能说明“工程链路可行”，不能说明 Orca 已具备抓取能力。
- 真正的 Orca cube grasp 仍需补简化碰撞体、cube 场景、接触奖励、lift/hold/success 指标。

## Aero-Hand 展示脚本

已扩展 `Aero-Hand/view.py`，新增 Orca 场景：

- `orca_right`
- `orca_left`
- `orca_combined`
- `orca_right_mjx_pose`

可视化命令：

```bash
cd Aero-Hand
/home/ll/miniconda3/envs/aero_rl/bin/python view.py --scene orca_right --mode overlay --preview-only
```

已生成预览图：

- `Aero-Hand/v2_iteration_docs/scene_previews/orca_right_visual_preview.png`
- `Aero-Hand/v2_iteration_docs/scene_previews/orca_right_collision_preview.png`
- `Aero-Hand/v2_iteration_docs/scene_previews/orca_right_overlay_preview.png`

其中 `visual` 显示外观层，`collision` 显示碰撞层，`overlay` 叠加显示，便于检查模型结构和碰撞几何是否对应。

## 当前限制

- Orca 原始 mesh 过重，不能直接作为大规模 PPO 碰撞模型。
- `scene_right_mjx_pose.xml` 没有碰撞体，只适合姿态跟踪和 DexMV 先验接通。
- DexMV 到 Orca 的映射目前是启发式：21 点手部关节映射到 Orca 17 个 actuator，尚未做指尖 IK 或接触点优化。
- 下一步要做 Orca cube 抓取，需要为每个手指建立简化碰撞体，并重新设计 cube 位置、接触奖励、lift/hold/success 指标。
