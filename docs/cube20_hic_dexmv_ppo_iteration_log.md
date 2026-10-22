# HIC 方块演示到 DexMV / Residual PPO 实验日志

## 目标与判据

目标是使用真实方块视频示教，经 DexMV 训练后桥接 Aero PPO，形成连续无支撑抓握超过 20 秒的策略，并导出视频。不能用旧 mustard 先验或累计接触时长代替本目标。

最终评测逐回合报告连续保持时长、支撑接触、手指接触、物体高度、掉落和种子；回放视频来自对应的新策略和配置。H03-holdout 已达到本次>20秒目标，最终结果与局限见文末。

## 数据来源

- HIC sequence 20: Moving a Cube with one hand。
- [官方数据入口](https://hic.is.tue.mpg.de/download.html)
- [格式说明](https://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/Dataset/Dataset_READ_ME.pdf)
- 本地目录：`artifacts/cube20/hic20/`。
- 原视频：`videos/20/input_rgb.mp4`，ffprobe 确认 30 FPS、186 帧、6.2 秒。
- 使用作者公开的跟踪运动拟合结果，**没有重新运行原始 RGB 姿态估计器**。
- 方块 OFF 网格范围为各轴 [-26.5, 26.5] mm，即边长 53 mm。

## H00：姿态格式和初始相对位置检查

命令：`/home/ll/miniconda3/envs/aero_rl/bin/python scripts/prepare_hic_cube_demo.py --fps 30`

数据包含 187 个运动帧；第 0 帧是绑定参考姿态，必须去除，运动帧 1 对应视频帧 0。将毫米转换为米，并按腕、拇指、食指、中指、无名指、小指生成 21 点。

初次预览发现方块和手指相差十几厘米。原因是物体骨骼原点不是网格中心：骨骼参考帧原点位于 x=-147.205 mm，而 OFF 网格中心位于零点。修正为刚性蒙皮变换 `M(t) @ inverse(M(0))`，不能直接把骨骼平移当作方块中心。

修正后输出：`prepared/human_cube.npz`、`prepared/hand_pose/`、`prepared/object_pose/`、`prepared/source_geometry.png`、带 SHA256 的 `prepared/manifest.json`。

骨节连接最大误差 1.98e-6 m。此数验证矩阵/轴/单位解析，不代表视频跟踪绝对精度，也不代表机器人抓取成功。

下一项配置验证：使用 DexMV 原有 PositionOptimizer 对掌根、各指关节点、指尖作有界优化，输出误差与初始物体相对位置；不把人手骨骼旋转直接冒充 MANO 的坐标系。

## 旧结果的使用边界

- May19 D04 仅是成熟 Aero PPO + 10% mustard 先验的历史基线。
- 旧报告 `contact_duration_sec=32.493` 是累计三指接触时间，不等于连续无支撑保持时长。
- `hold_success=132.25` 是累计成功标志，不是成功率。
- 本次数据转换和重定向还不算 DexMV 策略训练；必须另有 BC/DAPG 训练日志、保存策略和仿真验证。

## H01：标准重定向、BC 训练、物理验证

执行 `scripts/retarget_hic_cube_to_dexmv.py`，使用 DexMV 原有 SLSQP 位置优化器，186 帧平均误差 0.00959 m，指尖平均 0.00988 m，最大点误差 0.03626 m，关节越界为 0。为允许完整手部运动扩大 6 个根关节范围，手指限位未改。

执行 `scripts/train_hic_dexmv_bc.py`，使用 mjrl BC 的网络、归一化、MSE 损失和 Adam 优化器，600 轮。输入是时间相位，输出是 Adroit 姿态，是开环先验模型，不是闭环抓取策略。每五帧留一帧作验证，仅验证同片段插值，不代表新视频泛化。

| 指标 | first | last | min / best_step |
| --- | ---: | ---: | ---: |
| validation MSE | 0.012544 | 0.00024267 | 0.00012934 / 587 |
| validation finger RMSE(rad) | 0.08406 | 0.009985 | 最优 MSE 模型为 0.006900 |

模型：`artifacts/cube20/hic20/dexmv_bc/policy.pickle`。学习后的轨迹按真实 0.05 秒采样，而不是把 30 FPS 数据直接标成 20 FPS。

执行 `scripts/validate_hic_dexmv_bc.py`：从视频第 95 帧附近的学习姿态初始化，使用原 Adroit affine actuator 逆变换控制，重力开启，1 秒后撤支撑，模拟 6 秒。手指角度跟踪误差 first=0.000341 rad、last=0.03601 rad；连续无支撑保持最大 0 秒。**BC 学会了姿态，不等于独立抓稳。**

下一轮是环境/几何配置：只改重定向接触目标，将演示接近方块时的拇指/食指尖投向方块表面外 1 mm，增加这两个目标点权重；不改变 BC 网络或 PPO 奖励。原产物保持不动，写入 `dexmv_retarget_contact` / `dexmv_bc_contact`。

另已生成原始 BC 桥接数据 `ppo_bridge/demo_dataset.npz`。严格 PPO 评估的首次运行停在旧 Brax 配置读取问题，没有产生抓取结论；兼容修复仅在评估脚本中完成，没有改旧权重。

## H01-PPO：新视频先验 + 历史 PPO 的独立基线

脚本：`scripts/evaluate_hic_residual_hold.py`，无自动重置，800 步、dt=0.05 秒。逐步检查有效状态、支撑已撤、无支撑接触、无掌心接触、拇指加食指/中指至少一个指尖接触、方块高于初始位置减 1 cm。连续满足时间严格大于 20 秒才计成功。

使用 May19 D04 权重、新 HIC BC 的 `ppo_bridge/env_config.json`，没有在本轮更新 PPO 权重。种子 100/101/102/103 的连续时长分别 37.6/8.7/38.5/37.7 秒；3/4 达到阈值，种子 101 掉落。所有有效轨迹支撑释放后的支撑接触和掌心接触步数都是 0。

证据：`artifacts/cube20/hic20/eval_H01_initial/episodes.csv`、`trajectories.npz`、`report.json`。这是新先验的可行性正样本，不证明视频优于纯 PPO，也不是新先验下的训练终点。

下一轮类型：PPO 训练超参数/续训。保持环境、先验权重 0.1、奖励和初始几何不变，lr=1e-5、entropy=1e-4，从 D04 续训 81920 步（训练器可能按 batch 取整），64 个训练环境、16 个评估环境，episode=800。suffix=`HIC20_H02_bc_residual_cont81k`。训练完对新 checkpoint 做独立多种子评估，不以训练累计诊断项直接判成功。

## H01-contact：接触目标单变量对照

重定向命令增加 `--contact-weight 5 --output artifacts/cube20/hic20/dexmv_retarget_contact`，BC 参数不变，输出 `dexmv_bc_contact`。原演示目标点误差均值 0.00996 m，关节越界 0。

BC 验证 MSE first=0.012664、last=0.00016111、min=0.00013067（epoch559）；最优 MSE 模型的手指 RMSE=0.007255 rad。

物理检查初始接触由 1 指变为 2 指，末尾物体 z 从原版 -1.9637 m 改为 0.18585 m，但连续两指无支撑保持最大仅 0.006 秒，末尾仅 1 指被识别接触。因此不是已成功双指抓取，不据此替换 PPO 主线。主线继续使用 H01 的标准重定向 BC 先验。

该对照说明接触位置/手模型几何会影响结果，但没有证明单靠视频 BC 已经学会动力学抓取。最终闭环纠偏仍由 Aero PPO 完成。

## H02：新方块先验下 PPO 续训和 16 种子验证

运行目录：`Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbrResidualPriorAeroDirect-20261002-093308-HIC20_H02_bc_residual_cont81k`。

实际 num_minibatches=32、batch_size=256、unroll_length=40，一个采样批次为 327680 步；原请求 81920 步在 3 次评估设置下实际执行 655360 步，不能仍声称只训了 81k。JIT 约 47 秒，训练约 270 秒。

| 训练累计诊断 | first(step0) | last(step655360) | max | 推荐保存点 |
| --- | ---: | ---: | ---: | ---: |
| reward | 19262.783 | 20879.676 | 22683.072 | 327680 |
| drop | 0.25 | 0.25 | 0.25 | 327680，min=0.125 |
| hold_success | 111.625 | 107.1875 | 131.6875 | 327680 |
| contact_duration_sec | 27.128 | 29.081 | 31.975 | 327680 |
| palm_contact | 46.75 | 0 | 46.75 | 后两次均为 0 |

注意累计 hold_success/contact 不是最终成功率/连续时长。因后半段回退，选择中间 checkpoint `000000327680`，没有默认取最后一步。下一步是独立验证而非盲目长训。

固定新种子 200--215，不自动重置，连续无支撑超过 20 秒成功 11/16=68.75%；所有成功回合连续时长 37.7--38.4 秒；全部回合平均 26.88125 秒。失败种子 203/204/211/212/214 的时长分别 6.8/0/5.1/0/0 秒，未隐藏。

成功回合的支撑释放后支撑接触步数为 0，掌心接触步数为 0；失败种子 204/212/214 各有 1 步撤离过渡期支撑接触，且没有被计为成功。独立有效时间掩码把任何非有限/终止后的状态排除。

证据目录：`artifacts/cube20/hic20/eval_H02_16seeds`。视频 `best_rollout.mp4` 是 seed200，40 秒、20 FPS、800 帧、640x480；22 秒处截图确认方块在三指间，画面和日志均无支撑/掌心接触。

下一项验证：同一 checkpoint、同一 16 个种子，只把 blend_alpha 从 0.1 设为 0，进行运行时先验消融。它不是重新从零训练的纯 PPO，对其结果只作有限解释。

## H02-ablation：去掉运行时先验反而更稳

同 checkpoint、种子 200--215、同严格判据，仅 blend_alpha=0。结果 16/16 达标，平均连续保持 37.934375 秒，最长 38.4 秒；比 0.1 先验的 11/16、26.88125 秒更好。

证据：`artifacts/cube20/hic20/eval_H02_no_prior`。因此不能写“融合提升了成功率”，当前证据反而表明原始跨手型映射偏置在这一任务中有害。它仍不是完整重新训练的纯 PPO baseline，也不能推断所有视频先验都无效。

下一轮类型：环境/控制组合配置，仅把 blend_alpha 从 0.1 改为 0.05，保留非零先验；不改网络权重、奖励、初始姿态或评测判据。运行同一组种子，若正向，再用新的种子独立复核。原 10% 版本完整存档，不覆盖。

## H03：弱化错误先验偏置，保留残差融合

唯一变量 blend_alpha=0.05。同开发种子 200--215，成功从 11/16 提高至 15/16，平均连续保持从 26.88125 秒提高至 35.553125 秒，最长仍为 38.4 秒；失败种子207保持0秒。权重并未再训练，故应称“同一续训模型的控制组合参数修正”。

仍弱于本组运行时无先验的16/16，不能宣称视频提高成功率。选择5%是保留视频先验参与同时降低映射不匹配的折中，不是强先验融合已经解决。

证据：`artifacts/cube20/hic20/eval_H03_blend005`。下一项验证使用从未参与调参的种子300--315，并导出对应视频，输出 `eval_H03_holdout`；评估结束前不把15/16开发结果当最终泛化指标。

## H03-holdout：最终验证与归档

固定5%配置后，种子300--315全部16回合达标，连续时长37.6--38.55秒，平均38.096875秒，无终止/掉落，支撑释放后支撑接触和掌心接触均为0。没有改回合时长、奖励或成功判据来获得这个结果。

最终归档：`artifacts/cube20/final_hic_bc_residual_ppo_5pct`，含 PPO checkpoint、BC policy、实际先验、完整配置、逐回合CSV、逐帧NPZ、40秒回放视频和SHA256清单。seed301的视频展示最长38.55秒连续保持。

达到了“新方块视频演示 -> 训练先验 -> 桥接PPO -> 最终模型连续保持>20秒 -> 视频”的本次工程目标。没有达到或宣称：DexMV BC单独物理抓稳、视频先验优于纯PPO、任意初始位置泛化、Orca抓取成功、硬件部署成功。

归档复核：直接使用 `_5pct/checkpoint`、`_5pct/env_config.json` 重新运行seed301，连续保持再次为38.55秒、无掉落/支撑/掌心接触。6项回归测试与归档SHA256均通过。全部训练和验证进程已经结束。
