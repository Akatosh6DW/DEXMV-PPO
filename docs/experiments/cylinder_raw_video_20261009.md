# 圆柱：从原始视频重做实验

## 要做什么

用户确认：整个视频到训练链路重做，不复用已有追踪姿态、校正参考、示范数据或策略权重。
本次不是继续优化上一版成功率。旧实验保留在 `artifacts/orca_cylinder_study/`，结果不混用。

状态：原始 RGB 识别、物体几何估计、Orca 重定向和 32 次示范回放已完成；
9 次新初始化训练、66 项正式评估和 4224 回合统计全部完成，共 5898240 个在线环境步。
配置及评分审计通过，另外独立重算了 444 条已保存的完整预览轨迹。

**本轮严格 30 秒成功率全部为 0%，没有训练出稳定达标策略。**
BC 后 PPO 有少量原始 15 秒成功：最终正常位置 1/192（0.52%），其余方法正常位置为 0%。
这不是残差方法的成功，也不能声称视频引导总体优于纯 PPO。
详见[完整统计、图表与结论](../../artifacts/orca_cylinder_raw_video_20261009/report/results.md)。
另见[26 段正式对照录像](../../artifacts/orca_cylinder_raw_video_20261009/videos/index.md)
和[数据、源码与三种方法新权重索引](../../artifacts/orca_cylinder_raw_video_20261009/README.md)。
录像固定使用正常位置测试的首个初态，覆盖全部训练种子，不筛选成功案例；
手心视角，外观与碰撞体各一版，已检查帧数和非空画面。

新输出统一放在 `artifacts/orca_cylinder_raw_video_20261009/`。

## 数据来源

- [DexMV 官方项目页](https://yzqin.github.io/dexmv/)。
- [原始番茄罐示范 41](https://yzqin.github.io/dexmv/dexmv_video_original/demonstration/relocate_tomato/relocation-tomato_soup_can-41-raw.mp4)。
- 本地文件：`source/tomato41_raw.mp4`，640 x 480，15 fps，102 帧，6.8 秒。
- 选择原因：官方公开原始 RGB 视频、物体为圆柱罐、包含接近与抓握过程。
  选择时没有进行仿真成功率筛选。不是使用官网姿态叠加视频或机器人渲染视频冒充原始视频。
- 画面检查：`source/contact_sheet.png`。实际动作含手臂移动；固定底座 Orca
  任务不能声称复现整条人类手臂搬运轨迹，需明确截取的抓持阶段和坐标映射。

## 链路与边界

1. 从 RGB 像素重新识别人手关键点，保存逐帧有效性、时间戳和叠加可视化。
2. 根据新识别结果重新映射到 Orca 关节。映射仅允许几何/运动学拟合，
   不根据模拟抓取是否成功再搜索关节偏置。
3. 仿真物体与视频中的圆柱罐对应。真实尺寸、坐标及遮挡误差必须记录；
   不把单目估计描述为准确的物体六维姿态真值。
4. 重新生成示范，记录全部尝试，不悄悄用旧成功示范填补失败。
5. 从新随机初始化训练纯 PPO、BC 后 PPO、视频参考加残差 PPO。
6. 同一任务、同一在线交互预算和配对测试初态评估，保留失败与全部种子。

允许复用通用代码、Orca 的机械几何/动力学模型和通用视觉预训练模型。
禁止复用旧任务的 `.MOTION`、参考 `.npz`、示范集和策略 `.pkl` 作为输入。
通用视觉预训练模型负责认出手，并非以前训练的抓取策略。

## 当前发现

`scripts/prepare_hic_cube_demo.py` 读取 HIC 已发布的 `.MOTION`，不是 RGB 识别。
本地 `DexMV/docs/demo_gen.md` 也明确假设已取得手和物体姿态，未包含原始视觉识别算法。
因此不能仅重跑旧转换脚本就称为从视频重做。

视觉识别在项目内独立 `vision_env` 配置，不改动 `aero_rl`。
使用 [MediaPipe 官方 Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
进行逐帧手部关键点提取。单目输出存在尺度和遮挡不确定性，必须先验证，不能直接当作准确示范。

### 实际完成与发现的问题

1. 视觉环境建立在新实验目录的 `vision_env/`，MediaPipe 0.10.21、OpenCV 4.11.0.86。
   102 帧均检测到手，使用 CPU 推理。这是检测覆盖率，不是 100% 姿态准确率。
2. 识别结果在 `perception/tomato41/`，`overlay.mp4` 可看到重新识别的关键点。
   实际是左手示范，右手 Orca 映射显式使用镜像，而不是把左右手混在一起。
3. `perception/can41_v2/` 是新罐口轮廓估计。第一版第 44 帧将皮肤与罐口合并，
   主动报错；修改白色区域分割后 102 帧通过。第一版部分输出保留作调试记录。
4. 没有标定/深度，采用弱透视近似：以罐口椭圆和已知罐体尺寸估计尺度、轴向，
   手部深度使用视觉模型预测；沿深度方向的整体偏移用假定指尖接触罐面的几何约束拟合。
   这不是六维物体姿态真值，也不是经过独立精度验证的完整 DexMV 感知系统复现。
   罐口分割、首帧区域和选帧范围针对这段视频设定，不是换任何视频都自动可用的通用识别器。
5. 圆柱按 DexMV 的 YCB 番茄罐碰撞 STL 包围尺寸构建，直径 69.644 mm、高 103.637 mm，
   不再沿用旧实验的 53 mm 方块替代形状。质量 35 g 是仿真任务假设，不是从视频测量或真实装满罐质量。
6. 仅提取视频 2–4 秒的手指闭合阶段，去掉人类手臂平移，用各帧掌部坐标对齐右手。
   根据关节范围做指尖位置逆运动学拟合，末帧之后保持参考目标，不伪称原视频有 40 秒抓持。
7. 重定向各指平均误差约 23–36 mm，95% 分位约 36–78 mm，说明这条直接映射质量不足。
   这些数是机械手指尖相对视觉估计目标点的拟合残差，不是相对真实三维手部的测量误差；本视频没有可用独立真值来给出后者。
   新参考文件是 `retarget_neutral/reference.npz`，场景是同目录 `model.xml`。
8. 所有方法以零关节目标中立起手。最初用首帧拟合手型初始化出现约 3 mm 穿透，
   因而在正式回放前改为统一中立起手；未改变参考动作、物体位置来追求成功。
9. 32 次示范回放没有一次有效抓持，均在约 0.4 秒终止；只留下 257 个有效样本，
   24 回合训练部分 193 个样本、8 回合验证部分 64 个样本。
   这是同一视频参考的不同扰动回放，不是 32 个独立视频，更不是 32 条成功专家示范。
10. 最初怀疑托板失效，逐步核查后发现托板有接触，参考早期动作将罐碰走。
    保留此失败，不增加夹具、不改变重力、不搜索成功手型。
11. 旧 BC 接口硬性要求晚期抓持样本，因此拒绝这批数据。第一次训练尝试已停止，
    权重未续用，记录在 `debug_v1/`。新接口均匀采样终止前真实观测/动作，
    不伪造晚期样本、不使用终止后的物理轨迹作为 BC 标签。协议 v2 记录了这项接口修正。
12. 12 项回归测试已通过，包括不完整矩阵禁止生成最终报告、零成功仍需给出统计不确定性。
    初始化文件读取审计只发现本次新参考，没有旧 `.npz`/`.pkl`/`.MOTION` 输入。
    GPU 驱动工作正常；隔离沙箱内 CPU 测试产生的 CUDA
    设备不可见日志不表示宿主机驱动损坏。

### 冻结的比较配置

- 三种训练路线：纯 PPO、BC 后 PPO、视频参考加残差 PPO。
- 每种三个新种子：51001、51002、51003；不读取旧任务权重或刚中断的权重。
- 每次 655360 个在线环境步，同时保留 163840 步的短预算结果。
- 三种方法共同学习率 0.0003、初始化标准差 0.3，BC 固定 4000 次更新；不根据成绩调参。
- 每次 40 秒；撤托后连续有效抓持 30 秒、最后仍有效、没有禁止接触和超过 3 mm 的撤托后穿透才算清洁成功。
- 正式测试位置扰动 XY 各 ±2 mm、±6 mm，以及正常位置加 0.15 N 水平扰动。
- 每项测试固定 64 个新初态；9 次训练、66 项评估，已完成 4224 次正式测试。
- “仅 BC”和“仅参考”作为额外对照，区分 PPO 的增量与参考本身的作用。
- 基础参考每一步参与残差控制，修正上限 0.08 rad，约 4.58 度。
  最终目标还受关节范围及每步变化率限制，不能把网络原始输出直接当作已执行关节角。
- 本场景沿用所选机械模型的 0.01 秒物理步长，控制步长 0.05 秒，即每次控制推进 5 个物理子步；
  每个子步检查接触和穿透。它不是旧圆柱实验同一物体、同一姿态、同一数值配置下的追加种子。

## 本次执行命令

以下是执行记录，脚本默认拒绝覆盖已有输出，不能重复执行来覆盖本次结果。

```bash
# 视觉识别使用项目内独立 Python，不需要切换/改写 aero_rl。
artifacts/orca_cylinder_raw_video_20261009/vision_env/bin/python scripts/extract_raw_video_hand.py \
  --video artifacts/orca_cylinder_raw_video_20261009/source/tomato41_raw.mp4 \
  --model artifacts/orca_cylinder_raw_video_20261009/perception/hand_landmarker.task \
  --output artifacts/orca_cylinder_raw_video_20261009/perception/tomato41

artifacts/orca_cylinder_raw_video_20261009/vision_env/bin/python scripts/extract_raw_can_geometry.py \
  --video artifacts/orca_cylinder_raw_video_20261009/source/tomato41_raw.mp4 \
  --hand artifacts/orca_cylinder_raw_video_20261009/perception/tomato41/hand_landmarks.npz \
  --output artifacts/orca_cylinder_raw_video_20261009/perception/can41_v2

conda activate aero_rl
python scripts/retarget_raw_can_orca.py \
  --packet artifacts/orca_cylinder_raw_video_20261009/perception/can41_v2/hand_object.npz \
  --hand-model artifacts/orca_collision_refit/G08_thumb_convex_candidate/cube.xml \
  --output artifacts/orca_cylinder_raw_video_20261009/retarget_neutral

python scripts/prepare_orca_raw_can_study.py
python scripts/run_orca_raw_can_job.py --job artifacts/orca_cylinder_raw_video_20261009/jobs/collect.json
python scripts/run_orca_raw_can_study.py --phase train --workers 2
python scripts/run_orca_raw_can_study.py --phase eval --workers 2
python scripts/report_orca_raw_can_study.py
MUJOCO_GL=egl python scripts/render_orca_raw_can_study.py
```

训练环境变量：`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、
`XLA_PYTHON_CLIENT_PREALLOCATE=false`、`XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`。
矩阵执行器自动设置这些值。运行需宿主 GPU 可见，不需要修改驱动。

### 这轮能回答什么

它能检查新 RGB 感知、单目物体估计、跨左右手几何映射、示范、BC/PPO/残差接口是否衔接，
并观察固定预算下的实际结果。它不能据此证明某算法总体优越：尤其当示范只包含失败初段时，
BC 失败不能等同于“有高质量视频示范时 BC 没用”，残差失败也不能等同于“残差架构没用”。
不将单一视频、近似单目姿态或 3 个训练种子的结果外推到实机和任意物体。

## 三种策略每一步究竟输出什么

### 纯 PPO

网络看机械手/物体当前状态和模拟触觉，输出 17 个归一化动作；动作被转换为 17 个关节目标。
这里控制的是目标角度，不是网络直接输出关节力矩；模型中的位置反馈执行器再根据目标与实际角度/速度产生受限力矩。
没有 BC、没有视频参考动作输入、没有模仿损失。它仍使用三种方法共同的场景物体位置，
不应把“共同任务定义来自视频对齐”误说成“纯 PPO 网络也在读取参考动作”。

### BC 后 PPO

先用这次重新生成的观测/动作样本拟合策略，再让同一个网络靠环境反馈进行 PPO 更新。
执行时还是网络独立输出完整 17 维动作，不再额外加一条视频动作，也没有持续 DAPG 损失。
本次 BC 只有失败回放的初段样本，因此“BC 已训练”不代表“BC 已学会抓取”。
验证集是同一视频参考在另一批扰动初态下的回放，不是未见过的视频。

### 视频参考加残差 PPO

执行时每一步查新 `reference.npz` 的基础关节目标，网络输出修正：

```text
requested_delta = 0.08 * clip(PPO_output, -1, 1)
desired_joint_target = clip(video_reference(t) + requested_delta, joint_limits)
executed_target = limit_change(desired_joint_target, previous_target, 0.10 rad/step)
```

这是真正的执行时残差合成。不是把误差逐步累加，也不是只用视频设置起手。
评估同时保存基础动作、请求修正、最终执行目标、实际执行目标与基础动作的差，便于核查。
视频离线提取后，部署执行的是参考查表与网络推理，不是每步重新运行视觉识别器。
这里的基础项是按时间查表的关节轨迹，不是已经学成的通用基础抓取策略。
换物体位置、尺寸或任务时，需要重新对齐轨迹，或另行设计根据当前状态生成基础动作的模型；本轮未实现后者。

三种 PPO 都使用相同奖励：前 3 秒接近/接触反馈、之后的有效抓持反馈和掉落惩罚。
实际验收还检查子步穿透等更严格条件，因此奖励变高并不保证清洁成功率提高。

代码调用 Brax 时的 `restore_params=params` 是其接收“本次新随机参数或本次新 BC 参数”的接口，
不是从旧 checkpoint 续训；训练任务若指定 `checkpoint` 会报错。只有正式评估允许读取本实验新生成的权重。

## 术语

- 重定向：把人手动作转换为机械手可实现的关节目标，不是照抄人手角度。
- BC：行为克隆，根据新示范中的观测与动作训练策略模仿。
- PPO：通过仿真交互、奖励和反馈更新策略。
- 残差 PPO：执行时每一步均为“视频基础关节目标 + 有界 PPO 修正”，
  不只是先模仿或设置一个起手姿势。
- 数据溯源：为视频、识别结果、参考、示范和权重分别记录哈希及父输入，
  使结果能够追溯到这次原始视频，而非旧参考。

## 中间数据里面是什么

| 文件 | 主要字段 | 不是什么 |
|---|---|---|
| `perception/tomato41/hand_landmarks.npz` | 102帧的21个手部点、图像归一化坐标、模型估计的手中心三维坐标、左右手、时间戳 | 不是标定后的精确三维轨迹 |
| `perception/can41_v2/hand_object.npz` | 罐口椭圆、弱透视尺度、罐轴、手物相对几何、选定帧号 | 不是全局相机六维位姿；每帧使用罐口中心作为相对参考 |
| `retarget_neutral/reference.npz` | 31帧17关节目标`ctrl`、从0开始的时间`time`、物体初始化、目标/拟合指尖及误差 | 不是已经训练的策略；末帧后由程序保持目标 |
| `demonstrations/trajectories.npz` | 32回合的观测`obs`、动作`action`、终止前有效掩码`active` | 不是32条成功抓取，也不是32段视频 |
| `train/*/step_*/params.pkl` | 本轮对应方法和种子的策略/价值网络参数 | 不是录像，也不包含所有环境与残差参考 |
| `eval/*/step_000000000/episodes.json` | 每个测试初态的成功、最长连续保持、掉落、穿透和接触评分 | 不是每帧仿真状态 |
| `eval/*/step_000000000/preview_trajectories.npz` | 每项前4个固定测试案例的逐帧状态，用于审计和渲染 | 不是该项全部64回合的全轨迹 |

只有对已知来源的 `.pkl` 使用 `pickle` 加载；不要加载不可信第三方 pickle 文件。
