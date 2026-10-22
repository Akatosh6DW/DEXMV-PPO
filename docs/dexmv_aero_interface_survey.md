# DexMV 与 Aero-Hand PPO 接口梳理

生成时间：2026-05-12

## 结论

DexMV 和 Aero-Hand 可以对接，但不能直接复用 DexMV 的动作或 PPO 训练代码。主要原因是两边的机器人模型、动作维度、仿真栈和演示数据语义不同：

- DexMV 使用 MuJoCo 2.0 / `mujoco-py` / Adroit Hand，动作是 30 维 Adroit actuator 归一化控制。
- Aero-Hand PPO 使用 MuJoCo 3.x / MJX / JAX / Brax PPO，当前重点任务 `AeroCanGraspV2Force` 是 6 维灵犀手 V2 actuator 残差控制。
- DexMV 的价值在于提供人手视频到机器人关节轨迹、物体位姿、演示 observation/action 的生成流程；Aero-Hand 需要把这些结果转换成自己的 6 维控制、46 维 policy state、姿态奖励或 BC/demo loss 所需格式。

建议桥接路线：

1. 先用 DexMV 的 retargeting / demo generation 思路得到人手关键点、物体位姿、Adroit retarget qpos。
2. 新增一层 `DexMV -> Aero-Hand` 映射，把人手或 Adroit 关节轨迹转换成 Aero V2 的 11 维 hand qpos / 6 维 motor target。
3. 在 Aero-Hand 环境中重放这些 Aero 轨迹，生成 `demo_dataset.npz`，字段至少包含 `obs/state`、`actions`、`qpos`、`object_pose`、`aero_hand_qpos`、`aero_motor_targets`。
4. 再改 Brax PPO 训练脚本，加入 BC 预训练、demo loss、姿态奖励或 residual PPO。

## DexMV 工程结构

根目录：`DexMV/`

```text
DexMV/
├── README.md
├── environment.yml
├── setup.py
├── docs/
│   ├── demo_gen.md
│   └── env.md
├── examples/
│   ├── train.py
│   ├── retarget_human_hand.py
│   ├── visualize_retargeting.py
│   ├── visualize_policy.py
│   ├── relocate_random_action.py
│   ├── pour_water_random_action.py
│   ├── place_inside_random_action.py
│   └── configs/
├── hand_imitation/
│   ├── env/
│   │   ├── environments/
│   │   ├── models/
│   │   └── utils/
│   ├── kinematics/
│   │   ├── demonstration/
│   │   ├── mjcf_parser/
│   │   ├── optimizer.py
│   │   └── retargeting_optimizer.py
│   └── misc/
└── pretrained_model/
```

核心模块：

- `hand_imitation/env/environments/base.py`
  - 定义 DexMV 环境基类 `MujocoEnv`。
  - 标准接口：`reset() -> observation`，`step(action) -> (observation, reward, done, info)`。
  - 支持 `dump()` / `pack()` 保存和恢复 `qpos/qvel/qacc`。
  - 支持 `dump_mujoco_model()` / `pack_mujoco_model()` 保存和恢复 `body_pos/body_quat/site_pos/site_quat`。

- `hand_imitation/env/environments/ycb_relocate_env.py`
  - 任务类：`YCBRelocate`。
  - 控制频率：100 Hz。
  - horizon：400，但 `ignore_done=True`。
  - action：按 actuator 数量生成，Adroit XML 中为 30 维，接口范围为 `[-1, 1]`。
  - observation：`qp[:30] + palm_pos-obj_pos + palm_pos-target_pos + obj_pos-target_pos`，合计 39 维。
  - 物体：YCB 对象，如 `mug`、`mustard_bottle`、`tomato_soup_can` 等。

- `hand_imitation/env/environments/mug_place_object_env.py`
  - 任务类：`MugPlaceObjectEnv`。
  - observation：`qp[:30] + palm_pos-obj_pos + palm_pos-mug_pos + obj_pos-mug_pos + obj_quat`，合计 43 维。

- `hand_imitation/env/environments/mug_pour_water_env.py`
  - 任务类：`WaterPouringEnv`。
  - observation：`qp[:30] + palm_pos-obj_pos + palm_pos-target_pos + obj_pos-target_pos + obj_quat`，合计 43 维。

- `hand_imitation/kinematics/retargeting_optimizer.py`
  - 核心 retargeting 类：
    - `PositionKinematicsRetargeting`
    - `ChainMatchingPositionKinematicsRetargeting`
  - 输入：人手关键点序列和手部 frame 序列。
  - 输出：机器人 `qpos` 序列，目标是 Adroit 关节空间。

- `hand_imitation/kinematics/demonstration/base.py`
  - 定义 `DemonstrationBase`。
  - `fetch_imitation_data()` 输出单步 imitation data：
    - `observations`
    - `rewards`
    - `sim_data`
    - `actions`
  - `compute_action()` 通过 MuJoCo inverse dynamics 计算 actuator input，并可按 `act_mid/act_rng` 归一化到 `[-1, 1]`。

- `hand_imitation/kinematics/demonstration/relocation_demo.py`
  - 定义 `RelocationDemonstration`。
  - `play_hand_object_seq(retarget_qpos_seq, object_pose_seq)` 将 retarget 后的手部轨迹和物体位姿转成 demonstration dict：
    - `observations`: `np.ndarray[T, obs_dim]`
    - `rewards`: `np.ndarray[T]`
    - `actions`: `np.ndarray[T, action_dim]`
    - `sim_data`: `list[dict(qpos, qvel, qacc)]`
    - `model_data`: `list[dict(body_pos, body_quat, site_pos, site_quat)]`

- `hand_imitation/misc/data_utils.py`
  - 读取人手关键点和物体位姿。
  - 负责 camera/world 坐标变换、缺失帧补齐、插值、minimum jerk 时间参数化。

- `examples/train.py`
  - 训练入口，但依赖另一个工程 `dexmv-learn`。
  - 支持 BC、DAPG、SOIL、DensityONPG、TRPO。
  - demonstration 文件格式是 `.pkl`，读取后 `pickle.load(f).values()` 得到轨迹列表。

## DexMV 输出接口

DexMV 可以提供三类输出：

1. Retargeting 输出
   - 文件：通常为 `.pkl`。
   - 内容：Adroit robot joint qpos 序列。
   - 来源：`examples/retarget_human_hand.py`。

2. 物体位姿序列
   - 文件：`.npy` 序列。
   - 内容：每帧 object pose，通常是 `4x4` 或 `3x4` 变换矩阵。
   - 经过 `data_utils.py` 可以转换到 world frame。

3. Demonstration 输出
   - 文件：通常汇总成 `.pkl`。
   - 单条轨迹包含：
     - `observations`
     - `actions`
     - `rewards`
     - `sim_data`
     - `model_data`
   - 这些数据是 Adroit/DexMV 环境语义，不是 Aero-Hand PPO 可直接消费的格式。

## Aero-Hand PPO 工程接口

根目录：`Aero-Hand/`

PPO 相关代码集中在：

```text
Aero-Hand/sim_rl/mujoco_playground/
├── learning/train_jax_ppo.py
├── mujoco_playground/config/manipulation_params.py
└── mujoco_playground/_src/manipulation/aero_hand/
    ├── base.py
    ├── aero_hand_constants.py
    ├── grasp_cube_v2_force.py
    ├── grasp_bottle_v2_force.py
    ├── grasp_can_v2_force.py
    └── xmls/
```

核心接口：

- `base.py`
  - `AeroHandEnv` 继承 `mjx_env.MjxEnv`。
  - `action_size` 来自 `self._mjx_model.nu`。
  - 环境使用 MuJoCo XML 加载后转换为 MJX model。

- `aero_hand_constants.py`
  - V2 灵犀手接口：
    - `V2_NQ = 11`
    - `V2_NV = 11`
    - `V2_NU = 6`
  - V2 关节顺序：
    - `right_index_mcp`
    - `right_index_pip`
    - `right_middle_mcp`
    - `right_middle_pip`
    - `right_ring_mcp`
    - `right_ring_pip`
    - `right_pinky_mcp`
    - `right_pinky_pip`
    - `right_thumb_cmc_abd`
    - `right_thumb_cmc_flex`
    - `right_thumb_mcp`
  - V2 actuator 顺序：
    - `hw_thumb_rot`
    - `hw_thumb_flex`
    - `hw_index`
    - `hw_middle`
    - `hw_ring`
    - `hw_pinky`

- `grasp_cube_v2_force.py`
  - `CubeGraspV2Force` 是 V2 force-aware 抓握基类。
  - `reset(rng)` 生成 `mjx_env.State`。
  - `step(state, action)` 输入 6 维 action，输出新的 `mjx_env.State`。
  - action 不是绝对关节角，而是围绕 `_default_ctrl` 的残差：
    - `motor_targets = _default_ctrl + action * action_scale`
    - 再经过关节范围和单步 delta clip。

- `grasp_bottle_v2_force.py`
  - `BottleGraspV2Force` 继承 `CubeGraspV2Force`，改成圆柱/瓶类抓握逻辑。

- `grasp_can_v2_force.py`
  - `CanGraspV2Force` 继承 `BottleGraspV2Force`。
  - 当前 can 任务默认 XML：`scene_mjx_grasp_can_330ml.xml`。
  - 当前 action scale：
    - `[0.07, 0.255, 0.22, 0.22, 0.24, 0.082]`
  - 当前 `_default_ctrl`：
    - `[1.18, 0.36, 0.74, 0.78, 0.72, 0.52]`
  - 当前 `_pre_grasp_pose` 是 11 维 V2 hand qpos，用作姿态先验和 reset 初始姿态。

- `mujoco_playground/_src/manipulation/__init__.py`
  - 注册环境名：
    - `AeroCubeGraspV2Force`
    - `AeroBottleGraspV2Force`
    - `AeroCanGraspV2Force`

- `mujoco_playground/config/manipulation_params.py`
  - 为 `AeroCanGraspV2Force` 设置 Brax PPO 超参。
  - actor 使用 `policy_obs_key="state"`。
  - critic 使用 `value_obs_key="privileged_state"`。

- `learning/train_jax_ppo.py`
  - 当前是纯 Brax PPO 训练入口。
  - 已支持加载 checkpoint、配置 policy/value hidden sizes、选择 obs key、domain randomization、渲染。
  - 当前没有 DexMV demo 文件读取、BC loss 或 demo loss 逻辑。

## Aero-Hand 当前观测与动作语义

以 V2 force grasp 系列为准：

动作：

- shape：`(6,)`
- 范围：PPO 输出通常来自 tanh distribution，语义上是 `[-1, 1]`。
- 顺序：
  - thumb rotation
  - thumb flex
  - index
  - middle
  - ring
  - pinky
- 转换：
  - `motor_targets = default_ctrl + action * action_scale`

policy state：

- 当前单帧 state 维度为 46。
- 若 `history_len=1`，`obs["state"].shape == (46,)`。
- 拼接内容：
  - 6D motor targets
  - 5D tactile force EMA
  - 6D last action
  - 3D palm-cube position error
  - 3D cube linear velocity
  - 4D cube quaternion
  - 15D fingertip-to-cube vector
  - 2D support release phase
  - 2D hold duration / force balance

privileged state：

- 由 policy state 加上更多仿真真值组成。
- checkpoint 里出现过 `privileged_state.shape == 102`，`state.shape == 46`。
- 训练时通常 actor 看 `state`，critic 看 `privileged_state`。

环境输出：

- `reset(rng) -> mjx_env.State`
- `step(state, action) -> mjx_env.State`
- `State` 包含：
  - `data`: MJX qpos/qvel/ctrl/contact 等仿真数据
  - `obs`: `{"state": ..., "privileged_state": ...}`
  - `reward`
  - `done`
  - `metrics`
  - `info`

## 两边接口差异

| 项目 | DexMV | Aero-Hand PPO |
|---|---|---|
| 仿真栈 | MuJoCo 2.0 + `mujoco-py` | MuJoCo 3.x + MJX |
| RL 框架 | `dexmv-learn` / `mjrl`，BC/DAPG/TRPO/SOIL | Brax PPO / JAX |
| 机器人手 | Adroit Hand | Aero/Lingxi V2 hand |
| action 维度 | 30 | 6 |
| hand qpos 维度 | 30 手部/机械臂相关观测，XML 总 qpos 更大 | 11 hand qpos + object freejoint |
| observation 维度 | relocate 39，place/pour 43 | policy state 46，privileged 约 102 |
| demo 格式 | `.pkl` dict/list，含 obs/actions/sim_data/model_data | 当前无 demo loader，需要新增 `.npz` loader |
| 物体 | YCB mug/can/bottle/clamp 等 | cube/bottle/can 专用 XML |
| 控制语义 | 绝对 actuator command，经 `act_mid + action * act_rng` | `default_ctrl + residual_action * action_scale` |

## 是否能对接

可以对接，但对接层必须解决以下问题：

1. 手型不一致
   - DexMV retargeting 输出 Adroit qpos。
   - Aero-Hand 需要 11 维 V2 qpos 或 6 维 motor target。
   - 不能直接拷贝 DexMV action。

2. 动作语义不一致
   - DexMV action 是 inverse dynamics 反推出的 30 维 Adroit actuator command。
   - Aero action 是 6 维残差控制。
   - 需要通过 Aero 的 actuator 顺序和被动耦合关系生成 6 维 demo action。

3. 观测语义不一致
   - DexMV observation 与 Aero policy state 字段完全不同。
   - 如果用于 BC/demo loss，应该在 Aero 环境中重放 demo 并调用 `_get_obs()` 生成 Aero 自己的 `obs["state"]`，而不是使用 DexMV observation。

4. 坐标系需要统一
   - DexMV 有 camera/world extrinsic 处理。
   - Aero can/bottle XML 里物体位姿和抓握点固定在自己的 MuJoCo world 坐标。
   - 桥接脚本必须显式定义世界坐标映射、物体初始位姿和目标姿态。

5. 依赖环境不兼容
   - DexMV 需要 Python 3.7、`mujoco-py<2.1`、MuJoCo 2.0。
   - Aero-Hand 当前环境是 MuJoCo 3.5、MJX、Brax、JAX。
   - 建议不要强行在同一个 Python 环境中同时 import 两边；桥接可以用文件中间格式解耦。

## 推荐 `demo_dataset.npz` 中间格式

为了让 Aero-Hand PPO 后续最少改动，建议桥接脚本输出：

```text
demo_dataset.npz
├── observations/state           float32 [N, 46]
├── observations/privileged      float32 [N, P]，可选
├── actions                      float32 [N, 6]
├── rewards                      float32 [N]，可选
├── dones                        bool    [N]，可选
├── episode_starts               int32   [E]
├── aero_hand_qpos               float32 [N, 11]
├── aero_motor_targets           float32 [N, 6]
├── object_pos                   float32 [N, 3]
├── object_quat                  float32 [N, 4]
├── source_retarget_qpos         float32 [N, 30]，可选
└── metadata                     JSON string 或单独 `metadata.json`
```

其中 `actions` 可由下面公式近似得到：

```text
action = (aero_motor_targets - default_ctrl) / action_scale
```

然后 clip 到 `[-1, 1]`。如果后续做 residual PPO，`aero_motor_targets` 可作为 demo/base policy target，PPO 学 residual。

## 建议的桥接实现顺序

第一阶段：离线数据转换，不改 PPO

1. 新建 `bridges/dexmv_to_aero_dataset.py`。
2. 输入 DexMV retarget `.pkl` 和 object pose `.npy`。
3. 只生成 `aero_hand_qpos/object_pos/object_quat/aero_motor_targets/actions`，先不生成 Aero obs。
4. 写 shape checker，确认动作维度 6、qpos 维度 11、episode 分段正确。

第二阶段：Aero 环境重放，生成 PPO 可直接使用的 obs

1. 在 Aero-Hand 环境里增加 demo replay helper。
2. 对每帧设置 `data.qpos`、`data.qvel`、`ctrl`、物体 freejoint。
3. 调用 `_get_obs(data, info, obs_history)` 得到 `state/privileged_state`。
4. 输出完整 `demo_dataset.npz`。

第三阶段：训练集成

1. 在 `learning/train_jax_ppo.py` 或单独训练入口里加 demo loader。
2. 加 BC pretrain：监督 policy mean/action。
3. 加 demo loss：PPO 更新时混入 `MSE(policy_action, demo_action)`。
4. 加姿态奖励：将 `aero_hand_qpos` 作为 target pose prior。
5. 加 residual PPO：policy 输出 residual，demo/base action 作为参考动作。

## 当前风险点

- DexMV 的 `dexmv-sim` 只包含环境和 retargeting，完整训练依赖 `dexmv-learn`，当前主工程还没有接入 `dexmv-learn`。
- DexMV 原始 demo 下载自 Google Drive，当前本地未看到 `DexMV/demonstrations/`。
- Adroit 到 Aero V2 的手指映射需要单独标定，尤其是 thumb、ring/pinky、PIP 被动耦合。
- 如果目标是 can grasp，DexMV 的 `tomato_soup_can` 或 `mustard_bottle` 轨迹可参考，但物体尺寸/坐标/抓握姿态需要适配 Aero can XML。
- Aero PPO 训练栈当前没有 demo 入口；BC/demo loss 需要改 Brax PPO 训练流程或新增外层预训练脚本。

## 下一步建议

优先做一个小闭环，不要一开始改 PPO：

1. 选一个 DexMV relocate demo 或先用示例 retargeting 结果。
2. 写 Adroit qpos 到 Aero V2 qpos 的手工映射表。
3. 生成 1 条短轨迹的 `demo_dataset.npz`。
4. 写可视化/检查脚本，在 Aero can XML 里重放手和罐的位置。
5. 确认重放姿态合理后，再进入 BC/demo loss/residual PPO。
