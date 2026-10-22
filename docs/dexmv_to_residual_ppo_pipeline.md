# DexMV 到 Aero-Hand Residual PPO 全链路说明

本文档记录当前已经跑通的最小闭环：DexMV 从视频衍生手部姿态得到 Adroit 手先验，再桥接到 Aero-Hand，最后用 PPO 学习每一步的残差修正量。当前目标是证明链路能通，性能暂不作为结论。

## 1. 当前已跑通结果

一键命令：

```bash
./scripts/run_video_to_residual_ppo_smoke.sh
```

本次 smoke 使用 DexMV 自带示例：

- 视频衍生手部姿态目录：`DexMV/examples/retargeting_source/relocate_mustard_example_seq/hand_pose`
- 物体姿态目录：`DexMV/examples/retargeting_source/relocate_mustard_example_seq/object_pose`
- DexMV retarget 输出：`artifacts/dexmv_retarget/example_retargeting.npy`
- PPO 桥接数据集：`artifacts/ppo_bridge/demo_dataset.npz`
- Aero residual PPO 日志：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPrior-20260513-032718-dexmv_residual_smoke`
- 最终 smoke checkpoint：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPrior-20260513-032718-dexmv_residual_smoke/checkpoints/000000002048`

smoke PPO 只训练了 `2048` timesteps，`eval/episode_reward=48.348`。这个数只说明训练、评估、保存模型都成功，不代表策略已经有实用抓取性能。

## 2. 链路到底在做什么

当前链路分 4 层：

1. DexMV 层：把视频衍生的手部关键点/姿态 retarget 到 DexMV 的 Adroit 手，得到 30 维 `adroit_qpos` 序列。
2. 桥接层：把 Adroit 30 维姿态压缩成 Aero-Hand V2 的 6 维控制先验 `aero_prior_ctrl`，同时生成 11 维耦合关节姿态 `aero_prior_qpos` 供检查。
3. Aero-Hand 环境层：新增环境 `AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPrior`，每一步从 `demo_dataset.npz` 读取当前帧视频先验。
4. PPO 决策层：PPO 不直接输出完整姿态，而是输出残差动作，最终控制为：

```text
final_ctrl[t] = clip(video_prior_ctrl[t] + residual_scale * PPO(obs[t]) * action_scale)
```

这里 `video_prior_ctrl[t]` 来自 DexMV，`PPO(obs[t])` 来自 Aero-Hand 当前触觉、关节、物体状态观测。也就是说，最终每一步动作由“视频先验 + PPO 修正量”共同决定，不是只把 DexMV 当初始姿态。

## 3. demo_dataset.npz 内容

`scripts/bridge_dexmv_to_aero_demo.py` 会生成：

| key | shape | 用途 |
| --- | --- | --- |
| `adroit_qpos` | `(122, 30)` | DexMV retarget 后的 Adroit 手姿态 |
| `aero_prior_ctrl` | `(122, 6)` | Aero-Hand 6 维执行器先验，Residual PPO 实际读取它 |
| `aero_prior_qpos` | `(122, 11)` | Aero-Hand 11 维耦合关节姿态，用于检查和可视化 |
| `aero_prior_action` | `(122, 6)` | 把先验转成 PPO action 空间后的调试/BC 数据 |
| `object_poses` | `(122, 3, 4)` | DexMV 示例物体姿态，当前 smoke 未用于 reward |
| `metadata_json` | scalar string | 记录映射规则、关节名、来源路径 |

Aero-Hand V2 的 6 维 action/control 顺序为：

```text
[thumb_rot, thumb_flex, index, middle, ring, pinky]
```

DexMV Adroit 的主要映射规则：

- 食指：`FFJ2/FFJ1/FFJ0` 加权弯曲量
- 中指：`MFJ2/MFJ1/MFJ0` 加权弯曲量
- 无名指：`RFJ2/RFJ1/RFJ0` 加权弯曲量
- 小指：`LFJ2/LFJ1/LFJ0` 加权弯曲量
- 拇指旋转：`THJ4`
- 拇指弯曲：`THJ3/THJ1/THJ0` 加权弯曲量

这一步是“跨手型降维映射”，不是完美运动学等价。后续要提高性能，应优先优化这个映射和 Aero 手模型/任务对齐。

## 4. 代码改动位置

主工程新增：

- `scripts/bridge_dexmv_to_aero_demo.py`：DexMV retarget 输出到 PPO 数据集的桥接脚本。
- `scripts/run_video_to_residual_ppo_smoke.sh`：端到端 smoke 脚本。

Aero-Hand 子工程新增/修改：

- `grasp_cube_v2_force.py`：新增 residual prior 读取逻辑和新环境类。
- `manipulation/__init__.py`：注册 `AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPrior`。
- `config/manipulation_params.py`：让新环境使用 V2 PPO 参数。
- `aero_hand/base.py`：把本工程内的 V2 手部 STL mesh 加入 MuJoCo asset 字典。
- 若干 V2 XML：移除原来写死的 `/home/ll/SRTP/Aero-Hand/...` meshdir，避免主工程作为子模块时路径失效。

## 5. 当前融合方式的创新点和优点

当前方案的核心创新是：把视频提取出的动作/姿态先验变成目标硬件环境里的“每一步控制基底”，再让 PPO 只学习小残差。它比“先用 DexMV 初始化一次，再 PPO 自己探索”融合更深，因为先验贯穿整个 episode。

主要优点：

- 训练探索空间更小：PPO 从视频先验附近修正，而不是从随机动作开始摸索完整抓取。
- 能处理物理不匹配：DexMV retarget 只保证运动学像人手/Adroit，PPO 在 Aero-Hand MJX 环境中用接触、力、奖励修正动力学问题。
- 工程依赖隔离：DexMV 用 Python 3.7/MuJoCo 2.1 生成 npz，Aero-Hand 用 Python 3.11/JAX/MJX 训练 PPO，中间只交换数据文件。
- 部署动作是 Aero-Hand 控制量：最终控制输出面向 Aero 6 维执行器，而不是 Adroit 手。
- 可以做消融：固定先验、无先验 PPO、BC 初始化、DAPG/TRPO、Residual PPO 都能在同一个任务里比较。

注意：当前“最终决策模型”严格来说是 `PPO checkpoint + demo_dataset.npz + residual combiner`。如果部署时不想携带外部先验序列，需要再做策略蒸馏，或把 phase/prior 编进策略观测，让网络学会内化这个先验。

## 6. 其他结合方法对比

| 方法 | DexMV 数据怎么用 | 最终决策主体 | 优点 | 缺点 | 适合程度 |
| --- | --- | --- | --- | --- | --- |
| 纯 PPO | 不用 DexMV | PPO | 工程最简单，完全按 Aero 奖励优化 | 探索难，低自由度手容易卡在坏接触 | baseline 必做 |
| BC 预训练 + PPO | 用 DexMV/Aero demo 监督初始化策略 | PPO | 初期动作像 demo，训练更快 | PPO 后期可能忘掉 demo；demo 只影响初始化 | 适合作为增强 |
| DAPG/TRPO | demo 梯度混入 policy gradient | TRPO/DAPG policy | DexMV 原框架成熟，demo 引导强 | 算法栈老，和 Aero JAX PPO 不同；不是 residual 控制 | 可做对照组 |
| Pose/demo reward + PPO | demo 作为姿态奖励目标 | PPO | 容易接入，能鼓励像视频 | 仍可能只学“像”，不一定抓稳；权重难调 | 可与 residual 叠加 |
| 当前 Residual PPO | demo 每一步作为控制基底，PPO 输出修正量 | 视频先验 + PPO | 融合最深，先验持续参与决策，适合修正 sim/hand mismatch | 需要 phase/先验管理，先验质量影响上限 | 当前主线 |
| 多先验库 + gating | 多段视频先验，网络选择/插值 | gating + PPO residual | 泛化到多个物体/抓法 | 数据和训练复杂度更高 | 后续扩展 |
| 策略蒸馏 | 先训练 residual，再蒸馏成单网络 | 单 PPO/蒸馏策略 | 部署最干净，不依赖外部序列 | 会损失部分可解释性，需要额外训练 | 硬件部署前推荐 |

建议主线：先把当前 Residual PPO 跑稳，再加 BC 或姿态奖励做辅助，而不是一开始就把 DAPG/TRPO、PPO、BC 全混在一起。

## 7. 是否需要别的数据集

当前不需要别的数据集也能跑通工程链路，因为 DexMV 自带 mustard 示例已经足够做 smoke。

但要做有说服力的论文/项目结果，需要补数据：

- 与 Aero-Hand 任务目标一致的物体视频，例如方块、瓶、罐、杯等。
- 每个物体多条视频：不同初始相对位置、不同抓取方向、成功/失败边界样本。
- 尽量保留物体姿态轨迹，因为后面可以把物体相对手的 phase/pose 用进奖励或先验选择。
- 最好采集和 Aero-Hand 低自由度结构相近的抓法，否则 DexMV 从人手/Adroit 映射到 Aero 时会丢太多细节。

更换数据时改这些环境变量即可：

```bash
HAND_POSE_DIR=/path/to/hand_pose \
OBJECT_POSE_DIR=/path/to/object_pose \
./scripts/run_video_to_residual_ppo_smoke.sh
```

## 8. 评测指标怎么设计

不要只看总 reward。建议分层评测：

| 指标 | 含义 | 为什么重要 |
| --- | --- | --- |
| `success_rate` | 完成任务比例 | 最核心结果 |
| `lift_success_rate` | 是否把物体抬离支撑面 | 区分碰到物体和真正抓住 |
| `hold_duration` | 支撑释放后稳定保持多久 | 对真实部署更关键 |
| `drop_rate` | 掉落比例 | 抓取稳定性 |
| `three_finger_contact` | 拇指+主要手指接触质量 | 防止靠掌心/非目标接触作弊 |
| `normal_force_mean` / force balance | 接触力是否足够且均衡 | 判断是不是可靠力闭合 |
| `slip_event` | 物体相对手滑动 | 检测临界抓取 |
| `residual_norm` | PPO 修正量大小 | 判断先验是否好，残差越小通常先验越匹配 |
| `pose_tracking_error` | 最终姿态离视频先验多远 | 分析 PPO 是在微调还是完全推翻先验 |
| `energy/action_smoothness` | 动作变化和控制能量 | 硬件安全和寿命 |
| robustness score | 外力、摩擦、质量、重力扰动下成功率 | sim2real 需要 |

Aero-Hand 环境已有许多诊断项，例如 `diagnostic/success`、`diagnostic/lift_success`、`diagnostic/hold_success`、`diagnostic/slip_event`、`diagnostic/three_finger_contact`，后续应把这些从 `metrics.csv` 固定汇总到实验表。

## 9. 任务怎么设计

推荐按课程难度推进：

1. 无物体姿态跟随：只检查 DexMV->Aero 映射是否自然、不卡关节。
2. 支撑台上接触：物体不要求抬起，只要求手指接触正确区域。
3. 支撑释放前抓稳：支撑还在时形成拇指+食指/中指接触。
4. 短时抬起/离台：先要求 1-3 秒稳定。
5. 长时保持：逐步到 10 秒、20 秒、30 秒。
6. 扰动鲁棒性：随机摩擦、质量、外力、重力方向。
7. 泛化评测：不同物体、不同初始位姿、不同视频先验。
8. 硬件小步部署：低速、限幅、急停、力阈值保护。

每个阶段都应该比较至少 4 个 baseline：纯 PPO、BC+PPO、姿态奖励 PPO、Residual PPO。

## 10. 低自由度导致性能差怎么处理

Aero-Hand 6 维控制比 Adroit/人手自由度少，低自由度问题不能只靠 PPO 硬学。建议从 5 个方向解决：

1. 改桥接映射：把 Adroit 多关节不是简单平均，而是拟合 Aero 手可实现的 synergy。可以用最小化指尖位置误差/接触点误差来优化 6 维 ctrl。
2. 改任务和视频：选择 Aero 手天然能做的抓法，例如三指捏、包络抓、侧向夹持，避免要求独立 DIP/PIP 的精细动作。
3. 改残差范围：当前 `residual_scale=0.25` 是保守 smoke 设置。训练稳定后可以调到 `0.5` 或 `1.0`，同时放宽 `action_scale` 或针对 ring/pinky 调整局部限幅。
4. 改手模型和接触：低自由度手尤其依赖指腹几何、摩擦、软接触和掌心形状。接触模型不准时，策略会表现得像“自由度不够”。
5. 多先验/分阶段：一个视频先验不一定适合所有初始位姿。可以建立抓法库，让策略先选先验，再做 residual。

如果这些都不够，才说明硬件自由度本身限制了任务，应降低任务目标或调整机械结构。

## 11. 下一步建议

工程上下一步应做：

1. 给 `demo_dataset.npz` 加可视化脚本，直接画出 6 维 ctrl 和 11 维 qpos 随时间变化。
2. 在 Aero 环境里记录 `residual_norm`、`prior_step`、`pose_tracking_error` 到 metrics。
3. 跑 3 个短对照：纯 PPO、Residual PPO、Residual PPO + 姿态奖励。
4. 用同一 eval 脚本汇总 success/lift/hold/drop/slip/contact 指标。
5. 把当前 mustard 示例换成 Aero 目标物体视频，减少跨物体先验偏差。

## 12. 新增：直接换成 Aero-Hand 目标手型的路径

已新增第二条路径，避免 DexMV 官方 retarget 先输出 Adroit 30 维姿态：

```bash
./scripts/run_video_to_aero_direct_residual_ppo_smoke.sh
```

这条路径的输入仍然是 DexMV 的视频手部结果：

```text
DexMV/examples/retargeting_source/relocate_mustard_example_seq/hand_pose/joints_*.npy
```

但输出直接变成 Aero-Hand V2 的 6 维控制先验：

```text
artifacts/ppo_bridge/demo_dataset_aero_direct.npz
```

本次 smoke 结果：

- 环境名：`AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect`
- 日志：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260513-113608-dexmv_aero_direct_residual_smoke`
- checkpoint：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260513-113608-dexmv_aero_direct_residual_smoke/checkpoints/000000002048`
- timesteps：`2048`
- smoke eval reward：`39.71711730957031`

为什么不是直接改 DexMV 的 `retarget_human_hand.py` 里一行 XML？

DexMV 官方脚本写死了 Adroit：

```text
adroit/adroit_relocate.xml
THJ / FFJ / MFJ / RFJ / LFJ joint mapping
```

Aero-Hand V2 的关节名和自由度完全不同：

```text
[thumb_rot, thumb_flex, index, middle, ring, pinky]
```

因此直接把 Aero XML 塞进 DexMV 原来的 ChainMatching retarget 会缺少关节映射。当前实现采用等价工程路线：DexMV 负责从视频给出 21 个手部关节点，新的脚本 `scripts/retarget_dexmv_video_to_aero_prior.py` 直接把这些关节点映射到 Aero-Hand V2 的 6 维控制空间。这样目标手型已经是 PPO 任务里的手，而不是 Adroit。

两条路线现在可以对比：

| 路线 | 中间手型 | 先验文件 | PPO 环境 | 说明 |
| --- | --- | --- | --- | --- |
| `dexmv_prior_cross_hand_mapping` | Adroit 30D | `demo_dataset.npz` | `...ResidualPrior` | DexMV 官方 retarget 到 Adroit，再跨手型映射到 Aero |
| `aero_direct` | Aero-Hand V2 6D | `demo_dataset_aero_direct.npz` | `...ResidualPriorAeroDirect` | DexMV 视频关节点直接映射成 Aero 控制先验 |

## 13. 先验和 PPO 最终怎么揉在一起

现在真正执行的动作不是单独 PPO，也不是单独视频先验，而是：

```text
final_ctrl[t] = prior_ctrl[t] + residual_scale * PPO(obs[t]) * action_scale
```

其中：

- `prior_ctrl[t]`：视频先验，来自 `demo_dataset.npz` 或 `demo_dataset_aero_direct.npz`
- `PPO(obs[t])`：策略网络根据当前观测输出的 6 维残差
- `residual_scale`：残差幅度，当前 smoke 为 `0.25`
- `action_scale`：Aero-Hand 每个 actuator 的动作尺度
- `final_ctrl[t]`：最终发给 Aero-Hand MuJoCo 执行器的控制目标

这叫“外置先验 residual policy”。它的优点是结构清晰，先验可替换，容易做消融。缺点是部署时必须同时带上：

```text
PPO checkpoint + prior npz + residual combiner
```

更深的融合可以继续做三步：

1. 把 `phase`、`prior_ctrl[t]`、`prior_qpos[t]` 加进 PPO observation，让网络知道当前参考动作是什么。
2. 在 reward 里加入 `residual_norm` 或 `pose_tracking_error`，让 PPO 倾向于小修小补，而不是完全推翻先验。
3. 训练好 residual 系统后做蒸馏，用一个单网络拟合 `final_ctrl`。蒸馏后的单模型更方便部署，但如果运行时不再显式做 `prior + residual`，它就不再是严格 residual 架构，而是“被 residual 系统训练出来的普通策略”。

## 14. 哪种才算残差架构

严格说，只有满足下面公式的才算 residual：

```text
最终动作 = 基础策略/先验动作 + 学习到的修正量
```

在本项目里就是：

```text
final_ctrl = DexMV_video_prior_ctrl + PPO_residual_ctrl
```

因此：

- 当前 `...ResidualPrior` 是残差架构：Adroit 先验跨手型映射成 Aero 先验，PPO 输出修正量。
- 当前 `...ResidualPriorAeroDirect` 也是残差架构：DexMV 视频直接生成 Aero 先验，PPO 输出修正量。
- 只用 DexMV 姿态初始化 PPO，不是残差架构，因为先验只在 reset 或训练初期起作用。
- 只做 BC 预训练后 PPO，不是残差架构，因为最终动作是 PPO 单独输出。
- 只把 demo 做 reward，不是残差架构，因为 demo 没有作为动作基底参与控制。
- DAPG/TRPO 本身不是残差架构，它是把 demo 梯度混进 policy gradient；除非它的动作计算也写成 `base_action + learned_delta`。

所以你的目标“视频得到的策略和 PPO 共同决定每一步姿态”，最贴近的就是当前两条 residual prior 路线。区别在于旧路线多了一次 Adroit→Aero 跨手型映射，新路线直接把视频手部结果转成 Aero-Hand 控制先验，更符合“把 DexMV 手型换成 PPO 任务手型”的方向。
