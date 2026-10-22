# 核心源码：按一次动作的路径读懂工程

[总目录](README.md) · [完整报告](report.md) · [全部中文注释副本](core_source/README.md)

## 1. 先说明这些文件能不能直接运行

`core_source/` 是原文件的完整阅读副本，增加了中文说明，**不是新的运行目录**。原源码通过相对目录寻找模型、导入其他模块；直接运行阅读副本会导致路径不对。

训练、评估、开发修改仍使用主工程 `scripts/`。这样做是为了不破坏冻结实验的源码哈希，也避免产生两套逻辑不一致的训练实现。每个副本顶部有原始路径；打包脚本同时检查Python语法树和非注释词法序列，确保仅增加注释。

这不是把庞大工程全部复制过来。优先挑选20个负责数据、参考、BC、残差、物理、训练和评分的文件；辅助模型加载、传感器和Brax算法本体通过原文件导入，不假装该目录能独立替代工程。

## 2. 只读四个文件，先抓住主线

### 第一个：真正“视频加PPO”发生在哪里

[orca_residual_controls.py](core_source/orca_residual_controls.py) 的 `compose()`：

```python
residual = clip(action, -1, 1) * bound
desired = clip(base + residual, low, high)
```

`base`是参考关节目标；`action`是网络的17维输出；`bound`正式实验为0.08rad。结果是期望关节目标，尚未经过环境的变化速率限制。它没有把上一步残差继续积分进去。

同文件 `load_reference()` 读取NPZ里的`ctrl/time`，验证维度和限位，插值到每个控制时刻。后半段没有视频动作时，参考保持末帧。

### 第二个：网络给完整动作，还是只给修正

[orca_residual_env.py](core_source/orca_residual_env.py) 的 `step()`：

- `arm == "residual"`：从当前时间查参考，调用`compose()`，再转成父环境的归一化动作。
- `arm == "pure"`：动作直接映射到各关节允许的完整目标范围。
- 两路都进同一个父环境执行，继续限速、推进物理、计算任务判据。

所以两种网络输出维度相同，并不意味着动作含义相同。不能把纯PPO的权重当残差权重直接塞进去。

`_obs()`把参考与时间信息送入残差策略；`requested_residual`和`effective_residual`用于分别追踪请求与执行结果。

### 第三个：BC到底训练什么

[train_orca_matched.py](core_source/train_orca_matched.py) 的 `dataset_arrays()` 与 `fit_matched_bc()`：

1. 验证示范和当前手模型、参考、物理后端一致。
2. 按整条轨迹划分训练/验证，再分早期接近和后期保持样本。
3. 用示范动作监督策略均值，更新Actor参数；Critic不是在这一步学成抓握专家。
4. 返回预训练参数，后续可以仅做BC评估，也可以交给PPO继续训练。

文件后半段保留较早的通用运行入口；**当前S02/S04引用的是这些共享函数，不是说它的所有可选功能都在本次正式实验开启了。**

### 第四个：PPO实际在哪里被调用

[run_orca_s03_job.py](core_source/run_orca_s03_job.py) 的 `main()` 后半部分调用 `brax.training.agents.ppo.train.train()`。

S04没有单独训练器，它用此训练流程产生的S03早期检查点。`guided`先执行BC分支，然后将预训练参数作为PPO初始化；`residual`建立残差动作环境，再用同一个Brax PPO接口训练。

`callback()`在固定步数记录开发评估和权重；`progress()`记录PPO优化指标。日志中的`max/best_step`用于诊断，不代表正式评估会选择最高分检查点。

## 3. 从视频一路读到最终动作

| 顺序 | 中文注释文件 | 看什么、对应哪个产物 |
|---|---|---|
| 1 | [retarget_hic_cube_to_dexmv.py](core_source/retarget_hic_cube_to_dexmv.py) | DexMV位置优化器；生成Adroit拟合点、目标点、物体姿态 |
| 2 | [prepare_hic_orca_object.py](core_source/prepare_hic_orca_object.py) | 同序列网格/尺寸绑定，物体局部坐标转换 |
| 3 | [probe_hic53_orca.py](core_source/probe_hic53_orca.py) | 受约束IK、共同刚体变换、连续轨迹，O15参考的来源 |
| 4 | [probe_orca_cube_acquisition.py](core_source/probe_orca_cube_acquisition.py) | E01在仿真中搜索小偏置，不是PPO训练 |
| 5 | [train_orca_acquisition.py](core_source/train_orca_acquisition.py) | `--mode collect`参考回放、筛选，形成D01 |
| 6 | [train_orca_matched.py](core_source/train_orca_matched.py) | BC数据验证、分割、模仿损失和共享初始化 |
| 7 | [orca_residual_controls.py](core_source/orca_residual_controls.py) | 参考插值、请求修正、关节限幅 |
| 8 | [orca_residual_env.py](core_source/orca_residual_env.py) | 完整动作/残差两种执行语义 |
| 9 | [orca_benchmark_env.py](core_source/orca_benchmark_env.py) | 共同控制限制、托板撤离、有效抓握判据 |
| 10 | [orca_refit_env.py](core_source/orca_refit_env.py) | 替换历史奖励，真正生效的基础分量 |
| 11 | [orca_acquisition_env.py](core_source/orca_acquisition_env.py) | 前3秒接近/接触辅助奖励 |
| 12 | [orca_hic53_env.py](core_source/orca_hic53_env.py) | 53mm物体、起手和参考元数据的接口契约 |
| 13 | [orca_matched_env.py](core_source/orca_matched_env.py) | Actor信息限制，带噪声与延迟的模拟触觉 |
| 14 | [orca_native_physics.py](core_source/orca_native_physics.py) | JAX回调原生MuJoCo，完整物理状态保存 |
| 15 | [orca_suite_env.py](core_source/orca_suite_env.py) | 六条件与每5ms接触审查，零/半修正干预 |
| 16 | [run_orca_suite_job.py](core_source/run_orca_suite_job.py) | S02单任务训练、评估、清洁成功评分 |
| 17 | [run_orca_s03_job.py](core_source/run_orca_s03_job.py) | 调优配方的训练、固定步数权重保存 |
| 18 | [run_orca_s03_eval.py](core_source/run_orca_s03_eval.py) | S04复用的正式评估入口；校验权重来源 |
| 19 | [run_orca_s04.py](core_source/run_orca_s04.py) | 41项新初态评估、原定统计与可视化安排 |
| 20 | [report_orca_suite.py](core_source/report_orca_suite.py) | 聚合数据、bootstrap、配对检验与Holm校正 |

更早的 `bridge_dexmv_to_orca_trajectory.py` 含36/53缩放，是旧36mm任务，不是本报告主要结果的当前重定向入口。文件名看起来更像“桥接”，不代表应该优先用它复现53mm任务。

## 4. 为什么继承这么多层，应该怎样追一次step

当前主线由外到内大致如下：

```text
OrcaS03Grasp（S03/S04额外配方参数）
  → OrcaSuiteGrasp（扰动、残差增益、子步审查）
  → OrcaAcquisitionGrasp（前3秒奖励）
  → OrcaNativeMatchedGrasp（原生MuJoCo执行接口）
  → OrcaMatchedGrasp（观测及模拟触觉）
  → OrcaHic53Grasp（同源物体/姿态/参考绑定）
  → OrcaRefitGrasp（当前基础奖励）
  → OrcaResidualGrasp（参考+修正，或完整动作）
  → OrcaBenchmark（目标限速、物理推进、有效抓握判据）
  → Aero-Hand内的OrcaCubeGrasp（模型和基础辅助接口）
```

不是每一层都重新实现`step()`；没有实现的层沿用父类。动作先一路传下去，再带着新状态一路返回。`OrcaRefitGrasp`替换奖励，`OrcaAcquisitionGrasp`随后追加前期辅助奖励。因此只搜第一处`reward =`会读错最终奖励。

小型包装器原文件：[orca_s03_env.py](../../../scripts/orca_s03_env.py)。底层模型：[Orca grasp_cube.py](../../../Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/orca_hand/grasp_cube.py)。

## 5. 一次残差动作的具体数据形状

```text
观测 state: (123,)
    → Actor（128 → 128隐藏层）
    → 归一化动作: (17,)
时间step → 参考角 base: (17,)
    → compose(base, action, bound=.08)
    → desired_ctrl: (17,)，单位rad
    → 统一目标变化限制，每步最多0.1rad
    → MuJoCo data.ctrl，物理推进10×5ms
    → 新状态、接触力、奖励、是否终止
```

17个执行器的具体名字、顺序和限位由当前模型决定。不要仅因为数组都是17维就认为不同手型或不同动作接口兼容。

网络结构实际由原文件 [train_orca_comparison.py](../../../scripts/train_orca_comparison.py) 的 `network_factory()` 提供。该文件还包含历史demo loss功能，但本次S02/S04并未调用它的demo正则训练分支；不要把“文件里有”当成“本实验用了”。

## 6. 要改哪里才不会改错对象

| 想改变什么 | 主要入口 | 必须同时注意 |
|---|---|---|
| 视频或手型映射 | `prepare_hic_orca_object.py`、`probe_hic53_orca.py` | 物体单位、坐标、关节名称、碰撞和新的示范数据 |
| 参考轨迹 | 新版本NPZ及对应任务配置 | 不覆盖已冻结参考，否则现有权重/数据契约失效 |
| 残差幅度 | `OrcaResidualGrasp`的bound及上层配置传递 | 修改后重训/新评估，检查实际限速后的修正；不能只改图表标签 |
| 辅助奖励 | `OrcaAcquisitionGrasp`与作业shaping | 与主成功指标分开；奖励变好不代表达标 |
| BC策略 | `fit_matched_bc()`、训练器BC分支 | Actor/动作语义、示范分割、验证误差与实际成功率 |
| PPO超参数 | 新作业JSON与训练器 | 记录新实验版本，不回写旧协议和旧成绩 |
| 物理扰动/评分 | `OrcaSuiteGrasp`、`summarize()` | 训练与测试是否一致；门槛不能看完成绩才挑 |

`Aero-Hand/view.py`主要用于展示和交互检查，不是这轮BC或PPO训练入口。本次没有修改它或删除任何快捷键。

## 7. 注释和原文如何验证

在主工程目录运行：

```bash
conda activate aero_rl
python scripts/package_orca_experiment_reading.py
python -m unittest discover -s scripts -p 'test_orca_experiment_reading.py'
```

脚本只复制图表/表格、生成完整数据附表并为核心源码增加阅读注释，不运行仿真、不训练、不导入DexMV或MuJoCo依赖。会覆盖本目录中生成的副本，不覆盖原运行源码及人工编写的总报告。

`manifest.json`记录每个原文件与副本的哈希、注释数，以及是否与S04冻结源码匹配。准备阶段文件若不在S04冻结清单内，会明确标记；不据此假称它就是历史某次运行的精确版本。
