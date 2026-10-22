# Orca cube grasp iteration log

目标：使用现有 DexMV 视频数据生成 Orca 手先验姿态，再用 PPO 残差策略修正动作，完成 cube 抓取并在支撑释放后稳定保持 15s 以上。

成功判据：环境内 `diagnostic/hold_success=1`。该指标要求 cube 在支撑释放后仍高于掉落阈值，至少两个手指代理碰撞体与 cube 接触，cube 速度低于阈值，并连续满足 15s。

## D00: environment smoke

- 时间：2026-05-19
- 环境：`OrcaCubeGrasp`、`OrcaCubeGraspDexMVResidual`
- 目的：确认 MJX 模型、环境注册、reset/step、观测维度和 DexMV 先验读取都能工作。
- 结果：两个环境都能 reset/step。零动作推进 3 步时 reward 约 1.04/1.09，`contact_count=0`，`hold_time_sec=0`。
- 结论：基础管线可运行，下一步进入 PPO 探测。零动作未接触 cube，训练需要学会闭合和居中。

## D01: short PPO probe, DexMV residual scale 1.0

- 日志：`Aero-Hand/sim_rl/mujoco_playground/logs/OrcaCubeGraspDexMVResidual-20260519-015953-D01_orca_cube_dexmv_residual_40k`
- 训练：`40960` timesteps，`64` envs，`episode_length=120`。
- 首轮/末轮/最大指标：`eval/episode_reward` 2.621 -> 9.839，max 9.839；`contact_count` 1.50 -> 1.31，max 2.06；`two_finger_contact` 0.125 -> 0.188，max 0.375；`hold_time_sec` 始终 0；`hold_success` 始终 0；`drop` 始终 1。
- 分析：策略能接触 cube，但支撑释放后无法形成稳定夹持。DexMV 直接映射到 Orca 后闭合幅度偏弱，PPO 残差权限不足以快速补偿跨手型差异。
- 下一轮单变量改动：`OrcaCubeGraspDexMVResidual.residual_prior.residual_scale` 从 `1.0` 提高到 `3.0`。

## D02: short PPO probe, DexMV residual scale 3.0

- 日志：`Aero-Hand/sim_rl/mujoco_playground/logs/OrcaCubeGraspDexMVResidual-20260519-020300-D02_residual_scale3_40k`
- 训练：`40960` timesteps，`64` envs，`episode_length=120`。
- 首轮/末轮/最大指标：`eval/episode_reward` -11.285 -> -9.402，max -8.670；`contact_count` 2.625 -> 3.563，max 3.563；`two_finger_contact` 0.688 -> 0.750，max 0.813；`hold_time_sec` 始终 0；`hold_success` 始终 0；`support_released` 始终 0；`drop` 始终 1。
- 分析：增大残差权限显著增加了接触，但训练轨迹经常在支撑释放前就把 cube 碰掉或推走。下一步需要确认是 cube 初始支撑/几何不稳，还是训练动作过激。
- 下一轮单变量候选：如果脚本化 rollout 显示强闭合能稳定到释放，则保留残差权限并降低探索或延长预抓取；如果强闭合也不稳定，则先调 cube/support 初始几何。

## D02b: scripted reachability check

- 检查方式：用 `jax.lax.scan` 跑固定动作 rollout，不做 PPO 更新。
- 结果：`dexmv_zero` 在第 36 步掉落，支撑强制释放是第 40 步；`dexmv_close_target` 最大接触数 3.0，但仍在释放附近掉落；把释放时间延迟到 2.5/3.5/5.0s 也只能把掉落时间顺延，`max_hold` 仍为 0。
- 结论：当前 Orca proxy cube 场景的手指碰撞、cube 位置、支撑释放几何还不具备可靠可抓取性。它证明了 Orca-DexMV-PPO 工程链路可运行，但不适合作为本轮 15s cube 成功策略的最终载体。

## D03: Aero mature cube env, 100% DexMV base from C157 checkpoint

- 日志：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260519-021622-D03_aero_residual_from_C157_probe`
- 改动：切到成熟的 Aero V2 cube grasp 环境，恢复 C157 稳定 PPO checkpoint，但动作基底使用 `100% DexMV prior`。
- 结果：`contact_duration_sec=0.331s`，`hold_success=0`，`success=0`，`drop=1`。
- 分析：C157 checkpoint 学到的是“默认控制基底 + PPO 残差”。直接把基底换成 DexMV 姿态会改变动作坐标系，导致旧策略失效。

## D04: Aero mature cube env, blended DexMV base from C157 checkpoint

- 日志：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260519-021913-D04_aero_blend01_from_C157_eval`
- 视频：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260519-022122-D04_aero_blend01_policy_video_side_overlay/rollout0.mp4`
- collision debug：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260519-022122-D04_aero_blend01_policy_video_side_overlay/rollout0_collision_debug.mp4`
- 单变量设计改动：`base_ctrl = 0.9 * default_ctrl + 0.1 * dexmv_prior_ctrl`，`residual_scale=1.0`。PPO 输出仍是完整 residual action。
- 结果：`contact_duration_sec=32.493s`，超过 15s 目标；`hold_success=132.25`，`success=132.25`，`drop=0.125`，`palm_contact=0`，`slip_event=0`。
- 结论：本轮最终可行策略采用 Aero 成熟 cube PPO checkpoint + DexMV 视频先验偏置 + PPO residual 修正。该结果达到“cube 稳定抓取 15s 以上”的要求，但仍有约 12.5% 评估掉落率，后续优化应降低掉落率并提高 DexMV 先验权重。
