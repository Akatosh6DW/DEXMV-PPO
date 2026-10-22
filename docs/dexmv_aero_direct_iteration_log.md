# DexMV Aero-Direct Residual PPO Iteration Log

## D01: direct prior, residual_scale=0.25, 131k probe

- 目的：测试直接从 DexMV 视频手部关节点生成 Aero-Hand 6D prior 后，Residual PPO 是否能在 cube grasp 任务中形成有效抓取。
- 代码 / 配置：沿用 `AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect`，`residual_scale=0.25`。
- 数据：`artifacts/ppo_bridge/demo_dataset_aero_direct.npz`。
- 命令：`learning/train_jax_ppo.py --env_name=AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect --suffix=dexmv_aero_direct_D01_ep128_131k --num_timesteps=131072 --num_evals=3 --num_envs=128 --num_eval_envs=16 --episode_length=128 --unroll_length=16 --num_minibatches=4 --num_updates_per_batch=2 --batch_size=256 --learning_rate=3e-4 --entropy_cost=1e-2 --policy_hidden_layer_sizes=128,128 --value_hidden_layer_sizes=128,128 --num_videos=0 --use_wandb=false`
- checkpoint 来源：从零训练。
- smoke test：此前 direct env reset/step 已通过，prior_len=122。
- 真实指标：
  - `eval/episode_diagnostic/success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/lift_success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/hold_success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/three_finger_contact`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/drop`: first 1.0, last 1.0, max 1.0, best_step 0
  - `eval/episode_reward`: first -4.0941, last -2.0526, max -2.0182, best_step 65536
- 诊断：
  - `two_plus_primary_contact` 为 1.0，说明能产生部分主指接触。
  - `normal_force_mean` 从 0.377 到 0.410，有力信号。
  - `drop=1.0` 且 lift/hold 全 0，说明接触没有转化成可保持抓取。
- 与 smoke 对比：smoke reward 为 39.717，但 64 步短 episode 不暴露释放后掉落；D01 128 步评估显示释放后失败。
- 修改原因：当前 direct prior 的 thumb_rot 约 1.03-1.09，而 cube 任务原稳定 pregrasp 更接近 1.38；`residual_scale=0.25` 让 PPO 对 thumb_rot 每步最大修正只有约 0.03 rad，权限太小。
- 下一轮：只改环境参数 `residual_scale`，从 0.25 放大到 2.5，测试 PPO 是否能修正先验偏差。

## D02: direct prior, residual_scale=2.5, 131k probe

- 目的：验证 D01 失败是否主要由残差修正权限过小导致。
- 代码 / 配置改动：只改 `default_config_residual_prior_aero_direct().residual_prior.residual_scale`，从 `0.25` 改为 `2.5`。
- 数据：`artifacts/ppo_bridge/demo_dataset_aero_direct.npz`。
- 命令：`learning/train_jax_ppo.py --env_name=AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect --suffix=dexmv_aero_direct_D02_scale2p5_ep128_131k --num_timesteps=131072 --num_evals=3 --num_envs=128 --num_eval_envs=16 --episode_length=128 --unroll_length=16 --num_minibatches=4 --num_updates_per_batch=2 --batch_size=256 --learning_rate=3e-4 --entropy_cost=1e-2 --policy_hidden_layer_sizes=128,128 --value_hidden_layer_sizes=128,128 --num_videos=0 --use_wandb=false`
- checkpoint 来源：从零训练。
- smoke test：`residual_scale=2.5` 已由 env reset/step 验证。
- 真实指标：
  - `eval/episode_reward`: first 49.3063, last 533.9363, max 533.9363, best_step 131072
  - `eval/episode_diagnostic/success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/lift_success`: first 0.0, last 28.0625, max 28.0625, best_step 131072
  - `eval/episode_diagnostic/hold_success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/three_finger_contact`: first 1.5, last 35.875, max 35.875, best_step 131072
  - `eval/episode_diagnostic/drop`: first 1.0, last 0.6875, max 1.0, best_step 0
  - `eval/episode_diagnostic/contact_duration_sec`: first 0.0750, last 1.7937, max 1.7937, best_step 131072
- 与 D01 对比：D01 `lift_success=0`、`three_finger_contact=0`、`drop=1.0`；D02 形成可观三指接触和抬升信号，drop 降低。
- 修改原因：D02 是正样本，说明 direct prior 不是完全无效；关键在于 PPO 需要足够残差权限修正视频先验与 cube 任务之间的偏差。
- 实际效果：短训内形成接触/抬升，但 128 步评估无法满足 30 秒 hold 成功阈值。
- 下一轮：从 D02 checkpoint continuation，保持 `residual_scale=2.5`，只改训练/评估时长，将 `episode_length` 提到 800，测试长时释放后保持趋势。

## D03b: D02 continuation, episode_length=800, lr=3e-4

- 目的：验证 D02 的接触/抬升趋势在长 episode 下能否转化成保持能力。
- 代码 / 配置改动：相对 D02，保持 `residual_scale=2.5`，把 `episode_length` 从 128 提到 800；PPO 超参数保持 D02 的 `learning_rate=3e-4`。
- 命令：`learning/train_jax_ppo.py --env_name=AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect --suffix=dexmv_aero_direct_D03b_scale2p5_ep800_cont262k --load_checkpoint_path ...D02.../checkpoints/000000131072 --num_timesteps=262144 --num_evals=3 --num_envs=128 --num_eval_envs=16 --episode_length=800 --unroll_length=16 --num_minibatches=4 --num_updates_per_batch=2 --batch_size=256 --learning_rate=3e-4 --entropy_cost=1e-2 --policy_hidden_layer_sizes=128,128 --value_hidden_layer_sizes=128,128 --num_videos=0 --use_wandb=false`
- checkpoint 来源：D02 `000000131072`。
- 真实指标：
  - `eval/episode_reward`: first 4683.8301, last 834.9583, max 4683.8301, best_step 0
  - `eval/episode_diagnostic/success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/lift_success`: first 48.25, last 0.0, max 48.25, best_step 0
  - `eval/episode_diagnostic/hold_success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/three_finger_contact`: first 34.375, last 64.4375, max 64.4375, best_step 262144
  - `eval/episode_diagnostic/drop`: first 0.625, last 1.0, max 1.0, best_step 131072
- 诊断：
  - D02 checkpoint 在长 episode step 0 已有一定抬升和释放后保持 reward，但还没有达到 30 秒 hold。
  - 继续训练后 reward 和 lift 退化，drop 回到 1.0；接触时长增加但没有转化成稳定持握。
- 失败模式：正样本 continuation 退化，更像 PPO 续训学习率/更新幅度过大，而不是先验完全无效。
- 下一轮：不改环境和 reward，只把 continuation `learning_rate` 从 `3e-4` 降到 `1e-4`，从 D02 checkpoint 重新续训。

## D04: D02 continuation, episode_length=800, lr=1e-4

- 目的：验证 D03b 退化是否由 continuation 学习率过高导致。
- 代码 / 配置改动：相对 D03b，只改 PPO `learning_rate`，从 `3e-4` 降到 `1e-4`。
- 命令：`learning/train_jax_ppo.py --env_name=AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect --suffix=dexmv_aero_direct_D04_scale2p5_ep800_lr1e4_cont262k --load_checkpoint_path ...D02.../checkpoints/000000131072 --num_timesteps=262144 --num_evals=3 --num_envs=128 --num_eval_envs=16 --episode_length=800 --unroll_length=16 --num_minibatches=4 --num_updates_per_batch=2 --batch_size=256 --learning_rate=1e-4 --entropy_cost=1e-2 --policy_hidden_layer_sizes=128,128 --value_hidden_layer_sizes=128,128 --num_videos=0 --use_wandb=false`
- checkpoint 来源：D02 `000000131072`。
- 真实指标：
  - `eval/episode_reward`: first 4555.5732, last 2559.5557, max 4555.5732, best_step 0
  - `eval/episode_diagnostic/success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/lift_success`: first 51.0, last 52.75, max 52.75, best_step 262144
  - `eval/episode_diagnostic/hold_success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/three_finger_contact`: first 35.375, last 70.0625, max 70.0625, best_step 262144
  - `eval/episode_diagnostic/drop`: first 0.6875, last 0.8125, max 1.0, best_step 131072
- 诊断：
  - 低学习率减轻但没有消除退化；lift/contact 可继续提高，但 drop 仍偏高，hold 仍为 0。
  - 当前最好仍是 D02 checkpoint 在长 episode 的 step 0 评估，而不是续训后的 checkpoint。
- 下一轮：不改 PPO、不改 reward，只改 residual prior 时序：`loop=True` 改为 `loop=False`，让视频先验播放完后保持最后一帧，避免长时 hold 阶段不断重复视频动作。

## D05: D02 continuation, loop=False, episode_length=800, lr=1e-4

- 目的：验证长时 hold 失败是否由 122 帧视频先验循环播放导致。
- 代码 / 配置改动：相对 D04，只改 `residual_prior.loop`，从 `True` 改为 `False`；加载 D02 checkpoint 时使用 `--ignore_checkpoint_env_config=true`，确保当前环境配置生效。
- 命令：`learning/train_jax_ppo.py --env_name=AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect --suffix=dexmv_aero_direct_D05_loopfalse_ep800_lr1e4_cont262k --load_checkpoint_path ...D02.../checkpoints/000000131072 --ignore_checkpoint_env_config=true --num_timesteps=262144 --num_evals=3 --num_envs=128 --num_eval_envs=16 --episode_length=800 --unroll_length=16 --num_minibatches=4 --num_updates_per_batch=2 --batch_size=256 --learning_rate=1e-4 --entropy_cost=1e-2 --policy_hidden_layer_sizes=128,128 --value_hidden_layer_sizes=128,128 --num_videos=0 --use_wandb=false`
- checkpoint 来源：D02 `000000131072`。
- 真实指标：
  - `eval/episode_reward`: first 4661.2246, last 1607.0093, max 4661.2246, best_step 0
  - `eval/episode_diagnostic/success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/lift_success`: first 51.0625, last 17.5625, max 51.0625, best_step 0
  - `eval/episode_diagnostic/hold_success`: first 0.0, last 0.0, max 0.0, best_step 0
  - `eval/episode_diagnostic/three_finger_contact`: first 35.375, last 73.3125, max 73.3125, best_step 262144
  - `eval/episode_diagnostic/drop`: first 0.75, last 0.9375, max 1.0, best_step 131072
- 诊断：
  - `loop=False` 没有解决长时保持问题；续训后仍然从 D02 初始状态退化。
  - 当前最好 direct-mapping 策略仍是 D02 checkpoint，本轮证明 prior 循环不是主瓶颈。
- 结论：direct 映射可提供接触/抬升先验，但对于 cube 长时稳定抓取仍不足。主要瓶颈更可能是视频物体不是 cube、direct 关节点到 6D control 的映射过粗、以及续训阶段 reward/接触稳定性不足。

## Best Direct-Mapping Checkpoint and Rollout

- 当前 direct-mapping 最可信 checkpoint：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260513-115415-dexmv_aero_direct_D02_scale2p5_ep128_131k/checkpoints/000000131072`
- 可视化 rollout：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260513-121412-dexmv_aero_direct_best_D02_render/rollout0.mp4`
- 评估结论：
  - `residual_scale=0.25` 时，策略能产生少量接触/力信号，但 lift/success/hold 全为 0，说明直接映射先验本身不足以抓住 cube。
  - `residual_scale=2.5` 时，PPO 能利用 direct prior 学到三指接触和抬起趋势，D02 的 `lift_success` 达到 28.0625，`three_finger_contact` 达到 35.875，`drop` 从 1.0 降到 0.6875。
  - 长 episode continuation 的 D03b-D05 都没有把抬起转化为稳定保持，`hold_success` 和 `success` 仍为 0。
  - 因此直接映射不是完全无效，但它目前更像“接触/抬起提示”，还不是可靠的 cube 抓取策略。

## Why Direct Mapping Is Limited

- DexMV 当前视频先验来自具体演示视频，动作形态和物体形状强相关；如果源视频不是 cube 抓取，直接拿来引导 cube，先验会有明显任务偏差。
- DexMV 手部姿态到 Aero-Hand 6D 控制量的 direct retarget 是低维压缩，丢掉了很多手指独立运动、指尖位置和接触几何信息。
- Aero-Hand 的自由度更低，很多人手或 Adroit 手可以做出的包络抓取，在 Aero-Hand 上只能近似。
- 当前 residual PPO 虽然可以修正先验，但如果 prior 与目标任务差太远，PPO 需要同时学“纠偏”和“抓取”，训练难度会明显上升。
