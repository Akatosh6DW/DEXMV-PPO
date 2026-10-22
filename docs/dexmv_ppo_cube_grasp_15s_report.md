# DexMV + PPO cube grasp 15s report

> 历史报告勘误（2026-10-02）：下文把 `contact_duration_sec=32.493s` 解释为支撑释放后的连续保持时长并不准确。代码累计的是三指接触时间，未要求持续无支撑；`hold_success=132.25` 也不是成功率。旧数值和产物保留供追溯，但不能仅用该表证明 15s/20s 目标。新方块视频链路和逐回合严格评测请见 `docs/hic_cube_dexmv_residual_ppo_report.md`。

## 1. 最终结果

本轮跑通的最终策略使用：

- 手型和任务：Aero-Hand V2 cube grasp。
- 视频先验：`artifacts/ppo_bridge/demo_dataset_aero_direct.npz` 中的 `aero_prior_ctrl`。
- 决策模型：PPO policy network 输出 residual action。
- 最终 checkpoint：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260519-021913-D04_aero_blend01_from_C157_eval/checkpoints/000000010240`
- 指标日志：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260519-021913-D04_aero_blend01_from_C157_eval/metrics.csv`
- 可视化视频：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260519-022122-D04_aero_blend01_policy_video_side_overlay/rollout0.mp4`
- 碰撞调试视频：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260519-022122-D04_aero_blend01_policy_video_side_overlay/rollout0_collision_debug.mp4`
- 20s 截图：`visualizations/d04_aero_blend01_rollout0_t20s.png`

关键评估结果：

| 指标 | 数值 | 含义 |
|---|---:|---|
| `contact_duration_sec` | `32.493s` | 支撑释放后保持有效接触的平均时长，已超过 15s 目标 |
| `hold_success` | `132.25` | 30s hold 成功相关累计指标，非零表示存在长时成功轨迹 |
| `success` | `132.25` | 环境 success 指标，同样为非零 |
| `drop` | `0.125` | 约 12.5% 评估轨迹出现掉落，后续仍可优化 |
| `palm_contact` | `0.0` | 没有靠掌心作弊托住 cube |
| `slip_event` | `0.0` | 没检测到明显滑移事件 |

结论：当前策略已经达到“cube 抓取并保持 15s 以上”的可行性目标。它不是最终论文级最优策略，因为还有掉落率，但工程链路已经闭环。

## 2. 最终融合公式

最终环境是：

`AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect`

每一步先读取 DexMV 视频先验：

```text
prior_ctrl[t] = demo_dataset_aero_direct.npz["aero_prior_ctrl"][min(t, T - 1)]
```

然后构造控制基底：

```text
base_ctrl[t] = 0.9 * default_ctrl + 0.1 * prior_ctrl[t]
```

PPO policy 根据当前观测输出 residual action：

```text
residual[t] = policy(obs[t]) * action_scale * residual_scale
```

最终发给 MuJoCo/MJX actuator 的控制目标是：

```text
motor_target[t] = clip(base_ctrl[t] + residual[t], ctrl_low, ctrl_high)
```

本轮最终参数：

```text
blend_alpha = 0.10
residual_scale = 1.0
loop = False
```

`loop=False` 表示视频先验播放到最后一帧后保持最后一帧，不再循环做重复动作。这样长时 hold 阶段不会被视频周期动作扰动。

## 3. 为什么不用 100% DexMV base

我做了一个对照：

- D03：`base_ctrl = DexMV prior`，从 C157 稳定 PPO checkpoint 恢复。
- 结果：`contact_duration_sec=0.331s`，`drop=1`。

失败原因：C157 PPO checkpoint 原来是在“默认控制基底 + PPO action”这个动作坐标系里学出来的。如果突然把动作基底换成 100% DexMV 姿态，policy 输出的同一个 action 会变成完全不同的 motor target，等于把已学会的控制规律打乱。

因此最终采用更稳的实用融合：

```text
默认控制基底为主，DexMV 视频先验提供小权重姿态偏置，PPO 负责实时残差修正。
```

这仍然是 residual 架构，因为最终控制不是纯 PPO 直接给出，而是“先验基底 + PPO residual”。

## 4. DexMV 到 PPO 的数据链路

当前链路是：

```text
现有 DexMV 视频数据
  -> DexMV/retarget 脚本提取人手或源手姿态
  -> 转成 Aero-Hand 的 6D 控制先验
  -> 保存 demo_dataset_aero_direct.npz
  -> PPO 环境每步读取 prior_ctrl[t]
  -> PPO 输出 residual 修正量
  -> MuJoCo/MJX 执行 motor_target[t]
```

当前使用的数据文件：

`artifacts/ppo_bridge/demo_dataset_aero_direct.npz`

关键字段：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `dexmv_hand_joints` | `(122, 21, 3)` | DexMV 视频提取/重建出来的 21 个手部关键点 |
| `aero_prior_ctrl` | `(122, 6)` | 映射到 Aero-Hand 6 个控制量后的先验 |
| `aero_prior_qpos` | `(122, 11)` | 映射到 Aero-Hand 11 个关节位置后的姿态 |
| `object_poses` | `(122, 3, 4)` | 视频中的物体位姿信息 |
| `dt` | 标量 | 视频先验帧间隔 |

本轮没有重新训练 DexMV 自己的 DAPG/TRPO 策略，而是把 DexMV 作为“视频姿态先验提取器”。最终能抓住 cube 的策略主体是 PPO policy，但每一步动作基底都受 DexMV 先验影响。

## 5. PPO 训练和验证过程

使用的成熟初始 checkpoint：

`Aero-Hand/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbr-20260428-184050-C157_long_dr_lr5e5_upd3_xminus4mm_lower4mm_pregraspnoise010_smooththumbgeom_from_C124best_relmax2p45_postgrasp135_force72_capsule_bottlepalm_2048/checkpoints/000002949120`

这个 checkpoint 本身是 Aero cube 任务里的稳定 PPO 策略。D04 把它放进 DexMV-blended residual 环境中评估并保存为新的 checkpoint。

复现 D04 评估命令：

```bash
cd /home/ll/SRTP/DEXMV-PPO/Aero-Hand/sim_rl/mujoco_playground
PYTHONPATH=/home/ll/SRTP/DEXMV-PPO/Aero-Hand/sim_rl/mujoco_playground \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ll/miniconda3/envs/aero_rl/bin/python learning/train_jax_ppo.py \
  --env_name=AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect \
  --load_checkpoint_path=/home/ll/SRTP/DEXMV-PPO/Aero-Hand/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbr-20260428-184050-C157_long_dr_lr5e5_upd3_xminus4mm_lower4mm_pregraspnoise010_smooththumbgeom_from_C124best_relmax2p45_postgrasp135_force72_capsule_bottlepalm_2048/checkpoints/000002949120 \
  --ignore_checkpoint_env_config=true \
  --num_timesteps=8192 \
  --num_evals=1 \
  --num_envs=64 \
  --num_eval_envs=16 \
  --episode_length=800 \
  --learning_rate=1e-5 \
  --entropy_cost=1e-4 \
  --num_videos=0
```

复现视频命令：

```bash
cd /home/ll/SRTP/DEXMV-PPO/Aero-Hand/sim_rl/mujoco_playground
PYTHONPATH=/home/ll/SRTP/DEXMV-PPO/Aero-Hand/sim_rl/mujoco_playground \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ll/miniconda3/envs/aero_rl/bin/python learning/train_jax_ppo.py \
  --env_name=AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect \
  --load_checkpoint_path=/home/ll/SRTP/DEXMV-PPO/Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260519-021913-D04_aero_blend01_from_C157_eval/checkpoints/000000010240 \
  --play_only=true \
  --num_timesteps=0 \
  --num_videos=1 \
  --episode_length=800 \
  --camera=side \
  --render_mode=overlay \
  --render_collision_debug=true
```

## 6. Orca hand 原型状态

我也搭了 Orca cube grasp 原型：

- 环境：`OrcaCubeGrasp`、`OrcaCubeGraspDexMVResidual`
- 先验文件：`artifacts/ppo_bridge/demo_dataset_orca_direct.npz`
- 结果：工程能 reset/step，也能进入 PPO；短训能产生接触，但支撑释放后无法保持。

关键问题是当前 Orca proxy 碰撞和 cube/support 几何还不具备可靠夹持条件。脚本化检查显示即使用强闭合目标，cube 仍在释放附近掉落，`max_hold=0`。这不是 DexMV-PPO 链路本身的问题，而是 Orca 抓取场景还需要继续做碰撞代理、cube 位置、支撑释放和抓握姿态的几何调参。

因此本轮最终达标策略采用成熟 Aero cube 环境；Orca 分支保留为后续手型迁移方向。

## 7. 术语说明

| 术语 | 通俗解释 |
|---|---|
| DexMV | 从视频/演示中恢复手和物体运动的工程，可提供“人或源手大概怎么抓”的动作先验 |
| 视频先验 / prior | 视频里提取出来的参考动作，不是最终控制命令，只是给策略一个起点或偏置 |
| retargeting | 把源手姿态转换成目标机械手关节/控制量的过程 |
| PPO | 强化学习算法，策略通过试动作、拿奖励、更新参数来学会抓取 |
| residual action | 残差动作，表示“在先验动作基础上再修正多少” |
| residual PPO | PPO 不直接决定完整动作，而是决定对先验动作的修正量 |
| checkpoint | 训练保存下来的策略权重和配置，可用于继续训练或渲染 |
| support release | 训练开始时 cube 放在支撑台上，手抓住后支撑台撤走，用来测试是否真正抓住 |
| `contact_duration_sec` | 支撑释放后，手和 cube 保持有效接触的时间 |
| `hold_success` | 长时间稳定持握成功指标；当前环境默认按 30s 判据统计，比 15s 更严格 |
| `drop` | 掉落率或掉落事件指标，越低越好 |
| `palm_contact` | 掌心接触，通常代表可能用掌心托住作弊；本轮最终为 0 |
| `slip_event` | 滑移事件，表示 cube 相对手发生明显滑动 |

## 8. 后续优化方向

1. 把 `blend_alpha` 从 `0.10` 逐步扫到 `0.15/0.20/0.30`，观察是否还能保持 15s，同时提高视频先验参与度。
2. 从 D04 checkpoint 做小学习率 continuation，目标是把 `drop=0.125` 降到 0。
3. 采集或选择 cube 抓取更匹配的视频数据，而不是用泛化视频先验。
4. 对 Orca 分支继续做 proxy collision 和 cube 位姿调参，再把同样的 blended residual 方法迁移到 Orca。
5. 评测时固定记录 `contact_duration_sec`、`drop`、`palm_contact`、`slip_event`、`normal_force_mean` 和视频，避免只看 reward。
