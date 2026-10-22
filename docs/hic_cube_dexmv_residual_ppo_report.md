# HIC 方块视频 + DexMV BC + Residual PPO

## 当前状态

本文件记录新的方块视频链路，不把旧 mustard 瓶示例作为方块视频结果。

已完成下载、姿态转换、DexMV 重定向、BC 先验训练、Aero 桥接、PPO 续训、严格评测和视频生成。最终使用非零 5% 视频先验：未参与参数选择的种子 300--315 中，16/16 连续无支撑抓握超过 20 秒，范围 37.6--38.55 秒，平均 38.096875 秒；均无掉落、支撑释放后支撑接触或掌心接触。这是该固定场景、该组种子的结果，不代表任意初始位置成功率 100%。

最终产物目录：`artifacts/cube20/final_hic_bc_residual_ppo_5pct/`。

- `best_rollout.mp4`：40 秒、20 FPS，seed301，连续保持 38.55 秒。
- `checkpoint/`：实际续训至 step327680 的 PPO 权重。
- `demo_dataset.npz`：由训练后的 DexMV BC 模型导出的先验。
- `dexmv_bc_policy.pickle`：DexMV BC 网络。
- `env_config.json`：最终 5% 组合配置及初始位置/扰动参数。
- `episodes.csv`、`trajectories.npz`、`report.json`：全部 16 回合的逐回合/逐帧证据。
- `manifest.json`：来源、视频种子、结果和产物 SHA256。

**重要限制：无先验的同权重对照在开发种子上比 10% 先验更好。因此本次证明的是链路可用且能完成长时抓握，不是视频引导优于纯 PPO。**

使用的目标手是 Aero-Hand V2，不是 Orca。Orca 抓取分支没有因为本次实验而变成成功策略。

## 视频、数据与算法分别做什么

```text
HIC20 方块操作 RGB 视频 + 作者发布的配套运动跟踪结果
  -> 解析骨骼运动，去除参考帧，毫米转米，恢复手和方块相对位置
  -> DexMV PositionOptimizer：人手 21 点 -> Adroit 30 维姿态
  -> DexMV/mjrl BC：训练“时间相位 -> Adroit 参考姿态”的神经网络
  -> 在真实 20 Hz 时间点查询已训练 BC 模型
  -> 跨手型映射：Adroit 30 维 -> Aero 6 维执行器参考
  -> demo_dataset.npz
  -> Aero 仿真中每步读取参考，PPO 根据当前观测输出修正量
  -> 限幅、限速后的执行器目标 -> 物理仿真 -> 新观测
```

**本次确实进行了 BC 模型训练，不只是把原姿态文件改名。**但 BC 是动作先验学习，不是已经学会物理抓取的 DexMV 控制器。原始 RGB 视频识别没有在本机重新跑；使用的是作者公开的跟踪拟合结果。

DexMV 侧没有使用 DAPG 或 TRPO。这两者不是实现视频先验 + PPO 的必经步骤，也不是这次产物的训练来源。最终闭环控制包含一个视频动作参考和一个 PPO 残差网络。

## 数据来源与格式检查

- [HIC 数据入口](https://hic.is.tue.mpg.de/download.html)：sequence 20，Moving a Cube with one hand。
- [官方格式说明](https://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/Dataset/Dataset_READ_ME.pdf)。
- [官方查看器刚性蒙皮实现](https://github.com/cvlabbonn/hands_3d_motion_viewer/blob/master/src/animation_Transform.cpp)：采用当前骨骼变换乘参考帧变换的逆。
- 本地原视频：`artifacts/cube20/hic20/videos/20/input_rgb.mp4`。
- 视频 186 帧、30 FPS、6.2 秒；骨骼文件还多一帧绑定参考姿态。
- 物体是 53 mm 边长方块。运动文件中的物体骨骼原点不是方块中心，必须使用 `M(t) @ inverse(M(0))` 变换 OFF 网格，不能直接拿骨骼平移摆方块。
- `prepared/manifest.json` 记录来源、单位和原始 MOTION 文件 SHA256。

单条演示只覆盖有限抓法，不代表支持任意视频、物体或初始位置。

## 先验如何训练

重定向沿用 DexMV 的有界位置优化器，拟合掌根、各指关节点和指尖。扩大根部 6 个运动关节的范围，使整只手能跟随视频；没有放宽手指关节限位。标准版平均位置误差 9.59 mm、指尖平均 9.88 mm、关节越界为零。

BC 使用 mjrl 的 MLP、归一化、均方误差损失和 Adam 优化器：

- 输入：时间相位及其正弦/余弦编码，共 3 维。
- 输出：归一化的 Adroit 30 维姿态。
- 两层隐藏层，每层 128 单元；学习率 0.001，batch=32，600 轮，seed=20。
- 每五帧留一帧验证。它测试同一片段的插值能力，**不测试新视频泛化**。
- 最优验证模型在 epoch587，手指角度 RMSE 为 0.006900 rad，初始为 0.084061 rad。
- 保存模型：`artifacts/cube20/hic20/dexmv_bc/policy.pickle`。

独立 MuJoCo 2.1 检查表明，BC 姿态能被跟踪，但标准版不能独立抓稳自由方块。一个接触目标优化对照也没有形成可靠双指保持，因此没有用它替换主线。

## 如何桥接到 Aero

文件：`artifacts/cube20/hic20/ppo_bridge/demo_dataset.npz`。

它的 `aero_prior_ctrl` 为 64x6，顺序是拇指旋转、拇指弯曲、食指、中指、无名指、小指；`aero_prior_qpos` 为 64x11，是耦合后的关节姿态。

不是把源数据帧间隔直接改为 0.05 秒，而是在每个 0.05 秒时间点查询 BC 模型。参考使用接近并夹持阶段，几何选取的终点是视频第 95 帧附近；后续固定最后一个参考姿态，不播放视频后半段的放开动作。

源手和 Aero 并非运动学等价。映射保留加权手指屈曲等低维信息，**不传递源视频的手腕平移、旋转或方块位置**。Aero 的手、方块、支撑台初始几何沿用已验证的方块任务。

每一步采用：

```text
base = 0.95 * 默认控制姿态 + 0.05 * 视频参考
target = base + PPO(当前观测) * action_scale
执行器控制 = target 经过关节控制范围和变化速度限制
```

这属于带弱先验基底的残差架构。0.05 是混合系数，不是视频对性能的“贡献百分比”。在闭环运行中先验持续参与控制，但它自身只按时间推进，不能感知滑动；闭环修正来自 PPO。训练阶段使用 0.1，开发集对照后仅把部署/评测组合参数改为 0.05，没有声称又训练了一套 5% 权重。

## 初始位置与任务边界

- 方块初始中心：`[0.021, -0.065, 0.1463]` m。
- 支撑台初始位置：`[0.021, -0.065, 0.1308]` m。
- 方块位置抖动当前为零，关节观测噪声等扰动保留。
- 支撑随机释放时间为 1.7--2.7 秒，最长 3 秒强制释放；撤离有 0.5 秒过渡。
- 手使用场景已有 home 姿态，而不是把 HIC 世界坐标生搬到 Aero。
- 外力、等效重力倾斜等扰动仍启用，没有为了评测全部关闭。

这是固定手位、物体在已知位置上的闭合抓握与撤支撑保持任务，**不是机械臂从任意桌面位置接近、抓起、搬运方块**。目前也没有证明硬件部署成功。

## PPO 从哪里学

PPO 的训练样本来自 Aero 仿真中的实际交互：观测、残差动作、奖励、下一步状态。视频不提供最终的 PPO 动作标签；它通过每步控制基底影响探索和执行。

策略观测包括手的控制/关节相关信息、触觉、物体相对状态、历史动作和任务阶段等；模型保存配置的 policy state 为 46 维，训练 value 使用的 privileged state 为 102 维。奖励沿用现有接触、保持、掉落和控制稳定性设计，没有因本次演示重写奖励。

不是从零训练：从历史 D04 Aero 策略加载权重，在新 HIC 先验下以小学习率继续优化。历史能力主要来自既有 PPO，这必须在实验结论中保留。

续训 lr=1e-5、entropy=1e-4、64 个并行环境，40 秒 episode。因 batch=256、32 minibatches、unroll=40，一个完整采样批次为 327680 步；请求的 81920 步实际取整到 655360 步。中间 step327680 优于最后一步，因此选用中间保存点。训练 JIT 约47秒，后续训练约270秒。

实际运行目录：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20261002-093308-HIC20_H02_bc_residual_cont81k`。

## 严格评测与已有基线

独立脚本：`scripts/evaluate_hic_residual_hold.py`。不使用训练 wrapper 的累计 success 作为成功率，也不在掉落后自动重置续算。

每一仿真控制步同时要求：仍存活且数值有限、支撑已释放、方块不接触支撑台、不接触掌心、拇指和食指/中指至少一个指尖实际接触、方块高于初始高度减 1 cm。任何一步不满足就打断连续计时；严格超过 20 秒算成功。

初始 H01 基线：种子 100/101/102/103 连续时长分别 37.6/8.7/38.5/37.7 秒，即 3/4。完整证据在 `artifacts/cube20/hic20/eval_H01_initial/`，包含失败轨迹，不仅保存最好的一条。

这只说明新先验与原 PPO 能配合运行，不证明比纯 PPO 更好。原报告中的累计接触时间 32.493 秒不等于此处的连续无支撑时长。

后续完整对照结果：

| 配置 | 种子/用途 | >20s 回合数 | 平均连续保持 |
| --- | --- | ---: | ---: |
| 续训模型 + 10% BC 先验 | 200--215，开发 | 11/16 | 26.88125 s |
| 同权重去掉先验 | 200--215，运行时消融 | 16/16 | 37.934375 s |
| 同权重 + 5% BC 先验 | 200--215，开发 | 15/16 | 35.553125 s |
| 固定 5% 配置后复核 | 300--315，未参与调参 | 16/16 | 38.096875 s |

10% 版的失败种子是203/204/211/212/214；5%开发版失败种子207。这些都保留在各自 `eval_*` 目录，不能只看最后一组100%就忽略它们。

减少先验权重改善了开发组结果，支持“直接映射偏置不够适合目标手”的判断。无先验对照不是独立重新训练的纯 PPO，也不足以判断示教对样本效率的影响；严谨比较需要同预算、多训练种子的完整对照。

## 术语解释

| 术语 | 本工程里的含义 |
| --- | --- |
| 先验 | 视频给出的参考动作，提示大致怎么动 |
| 重定向 retargeting | 把人手动作近似转换成另一个机械手能做出的动作 |
| BC 行为克隆 | 用演示输入输出做监督学习，本次训练时间到姿态的映射 |
| MLP | 多层全连接神经网络，本次用来拟合动作参考 |
| RMSE | 均方根误差，数值越小表示预测越接近参考；不直接等于抓取质量 |
| PPO | 通过仿真试动作、奖励和策略更新学习控制的强化学习算法 |
| 残差 residual | 对参考动作再加多少修正 |
| 开环 / 闭环 | 开环只按参考推进；闭环会根据实际观测纠正动作 |
| DAPG / TRPO | DexMV 可用的示教增强策略梯度 / 信赖域策略优化，本次没有使用 |
| checkpoint | 保存的网络权重和训练配置，可继续训练或回放 |
| seed 种子 | 固定随机过程，用来复现实验和比较策略 |
| 消融 | 去掉一个组成部分进行比较，判断它是否真正有用 |

## 方法比较与后续优化

| 方法 | 视频如何参与 | 本次是否完成 |
| --- | --- | --- |
| 纯 PPO | 不参与 | 有历史模型；本次仅做同权重运行时去先验对照，未独立重训 |
| BC 初始化 PPO | 只帮助网络起步，之后可能遗忘 | 本次未用，BC 训练的是独立先验 |
| DAPG | 把示教项混入策略梯度 | 本次未用，也不等于动作残差 |
| 姿态奖励 PPO | 在奖励中鼓励接近视频 | 本次未额外增加 |
| BC 先验 + Residual PPO | 学到的先验每步参与，PPO 闭环修正 | 本次主线 |

当前优点是跨环境交换数据清晰、先验来源可追溯、最终执行量属于目标手，并能把视频近似动作与物理反馈分开。残差控制、BC 和跨手型重定向本身都不是新发明；创新主张需要依赖明确的新映射、接触约束或实验收益，不能仅因把工程接起来就认定算法创新。

提高低自由度手性能应优先针对目标手优化指尖/接触位置、选其能实现的抓法、使用多条视频、控制残差范围。不能期待手指角度简单压缩后完全保留人手灵巧性。后续评测应增加随机物体位置、摩擦/质量变化和多训练种子，并同时报告成功率、连续保持时长、掉落、力、滑移、动作平滑性和样本效率。

## 复现入口

在主工程目录依次执行，DexMV 命令需设置既有 MuJoCo 2.1 环境变量：

```bash
/home/ll/miniconda3/envs/aero_rl/bin/python scripts/prepare_hic_cube_demo.py --fps 30
export LD_LIBRARY_PATH=/home/ll/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=/home/ll/.mujoco/mujoco210
/home/ll/miniconda3/envs/dexmv/bin/python scripts/retarget_hic_cube_to_dexmv.py
/home/ll/miniconda3/envs/dexmv/bin/python scripts/train_hic_dexmv_bc.py
/home/ll/miniconda3/envs/aero_rl/bin/python scripts/bridge_hic_bc_to_aero.py
/home/ll/miniconda3/envs/aero_rl/bin/python scripts/test_hic_pipeline.py -v
```

PPO 训练通过新增 `--env_config_json` 指定本次数据，不覆盖旧默认数据路径。详细训练/验证过程见 `docs/cube20_hic_dexmv_ppo_iteration_log.md`。

本次 PPO 续训命令（保留实际运行参数，包括批次取整行为）：

```bash
ROOT=/home/ll/SRTP/DEXMV-PPO
BASE="$ROOT/Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20260519-021913-D04_aero_blend01_from_C157_eval/checkpoints/000000010240"
cd "$ROOT/Aero-Hand/sim_rl/mujoco_playground"
PYTHONPATH="$PWD" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/ll/miniconda3/envs/aero_rl/bin/python learning/train_jax_ppo.py \
  --env_name=AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect \
  --load_checkpoint_path="$BASE" \
  --env_config_json="$ROOT/artifacts/cube20/hic20/ppo_bridge/env_config.json" \
  --suffix=HIC20_H02_bc_residual_cont81k \
  --num_timesteps=81920 --num_evals=3 --num_envs=64 --num_eval_envs=16 \
  --episode_length=800 --learning_rate=1e-5 --entropy_cost=1e-4 \
  --batch_size=256 --num_minibatches=32 --unroll_length=40 \
  --num_updates_per_batch=4 --num_videos=0 --use_wandb=false
```

复核最终保存包并重新生成单个回放：

```bash
cd /home/ll/SRTP/DEXMV-PPO
BUNDLE="$PWD/artifacts/cube20/final_hic_bc_residual_ppo_5pct"
PYTHONPATH="$PWD/Aero-Hand/sim_rl/mujoco_playground" \
  /home/ll/miniconda3/envs/aero_rl/bin/python scripts/evaluate_hic_residual_hold.py \
  --checkpoint "$BUNDLE/checkpoint" \
  --env-config "$BUNDLE/env_config.json" \
  --output artifacts/cube20/replay_final_seed301 --seeds 301 --render
```

旧10%版本保存在 `artifacts/cube20/final_hic_bc_residual_ppo/`，不要与推荐的 `_5pct` 版本混淆。整个文件夹仍依赖现有 Aero 模型资产和 aero_rl 环境，不是可直接上硬件的单个网络。此次没有提交或推送 Git。

## 交付复核

- 6 项回归检查通过，覆盖参考帧变换、非法旋转/时间参数、先验来源、真实采样间隔、连续计数与归档哈希。
- 从最终归档的 checkpoint 和先验重新加载并执行上述 seed301 命令，再次获得38.55秒连续无支撑保持；结果在 `artifacts/cube20/replay_final_seed301/`。
- ffprobe 确认最终视频40秒、800帧、20 FPS、640x480；人工检查30秒处画面，方块位于三指间，支撑台不在物体下方。
- 数据转换、训练、评估均已结束，没有依赖仍在后台运行的任务来支撑结果声明。
