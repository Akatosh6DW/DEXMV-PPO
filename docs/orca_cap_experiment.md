# Orca 固定瓶身拧盖：开发记录

状态：已完成一对开发训练及独立复核；不是正式多种子方法对照，
也未完成完整旋出取盖。**最终成功率以 `audited_metrics.csv` 为准**，原始
`metrics.csv` 保留了修正前的在线计数，见 C15。
任务说明和公平性讨论见 [人话版实验设计](orca_experiment_explained.md)。

## C01：先验证物理可达性

- 新增 `scripts/orca_cap_scene.py`，保留 Orca V2 17 个执行器，使用独立被动瓶盖。
- 固定瓶身，瓶盖中心 [0.055, 0.025, 0.265] 米，轴向世界 +X。
- 瓶盖半径 19 毫米，厚 32 毫米，质量 15 克；螺距 2 毫米每圈。
- 程序逆运动学生成 2 秒接近、6 秒旋转 90 度目标并保持，非视频示范、非 PPO。
- 命令：`python scripts/orca_cap_scene.py --output artifacts/orca_cap/c01_x`。
- 没有初始策略权重；这是仿真几何 smoke test，无 PPO metrics.csv。
- 12 秒原生仿真：最大 53.018 度，末尾 52.253 度；有效拇指配对接触比例 85.42%，
  非手指接触比例为 0；最大接触力 18.90 N。
- **失败**：螺纹关系最大误差 10.249 毫米，不接受为有效拧盖。
- 零动作对照最大旋转约 1.35e-15 度，没有自行旋转。
- 下一轮类型：环境约束参数，只提高螺纹 equality impedance，检查误差能否降下来。

## C02：收紧螺纹约束

- 唯一主动改动：equality `solimp` 从 `.99 .999 .001` 改为 `.9999 .9999 .001`。
- 命令：`python scripts/orca_cap_scene.py --output artifacts/orca_cap/c02_stiff_thread`。
- 最大 53.674 度，末尾 53.583 度；最大螺纹误差仍有 3.128 毫米，不接受。
- 接触配对 85.42%，非手指接触为 0，零动作最大转角约 7.41e-15 度。
- 下一轮类型：环境求解器，单独把迭代次数从 8 增至 40，检查约束求解不足。

## C03：增加求解器迭代

- 唯一主动改动：MuJoCo `iterations=40`，其他设置同 C02。
- 命令：`python scripts/orca_cap_scene.py --output artifacts/orca_cap/c03_solver40`。
- 结果与 C02 完全相同；增加最大迭代次数无效，不重复加迭代。
- 下一轮类型：环境积分步长，从 10 毫秒降到 2 毫秒，控制周期仍为 50 毫秒。

## C04：减小物理积分步长

- 唯一主动改动：物理 dt=0.002 秒，每个控制步执行 25 次物理步。
- 命令：`python scripts/orca_cap_scene.py --output artifacts/orca_cap/c04_dt2ms`。
- 最大 71.987 度，末尾 62.732 度；螺纹误差 1.394 毫米，接触力峰值 52.00 N。
  虽然减小误差，仍大于四分之一圈的目标上升量 0.5 毫米，不接受。
- 下一轮类型：环境约束参数化；将相同螺距关系改写成角度等于轴向位移乘 3141.59，
  检查软约束量纲缩放的影响，不改变目标螺距。

## C05：螺纹关系的量纲缩放

- 关系由 `lift = pitch * turn` 改写为 `turn = lift / pitch`，仍为相同 2 毫米每圈。
- 命令：`python scripts/orca_cap_scene.py --output artifacts/orca_cap/c05_scaled_equality`。
- 最大 72.386 度，末尾 62.411 度，最大螺纹误差 0.523 毫米；没有非手指接触。
- 下一轮类型：环境约束时间常数，由 8 毫秒改为 4 毫秒（当前 dt 的两倍）。

## C06：约束时间常数

- 仅将 equality solref 改为 `.004 1`，其余同 C05。
- 命令：`python scripts/orca_cap_scene.py --output artifacts/orca_cap/c06_time_constant`。
- 最大 72.575 度，末尾 62.475 度，最大螺纹误差 0.327 毫米；配对接触 85.42%，
  非手指接触为 0。仍未达到 90 度，不标成功。
- 下一轮类型：先验证 MJX 环境，不把原生仿真程序控制结果当作 PPO 结果。

## C07：MJX 接口 smoke test

- 独立新增 `orca_cap_env.py` 与 `train_orca_cap.py`，不改 F01 冻结源码。
- 成功标准：转角至少 90 度，拇指加另一手指接触，连续保持 1 秒；保持时螺纹误差
  小于 0.2 毫米。全回合曾出现非手指接触或误差超过 1 毫米的回合不计成功。
- 12 秒回合、控制 dt=0.05 秒、物理 dt=0.002 秒；观测 95 维、critic 112 维。
- 奖励 v1：转角进度、接近、配对接触、有效保持；惩罚非手指接触、螺纹误差和动作突变。
  不给瓶盖施加控制力矩，不修改 qpos 制造转动。
- 命令：`python scripts/train_orca_cap.py --model artifacts/orca_cap/c06_time_constant/model.xml --output artifacts/orca_cap/c07_mjx_smoke --smoke`。
- 当前训练基线没有视频引导；后续跨任务视频先验必须另标来源，不能把 IK 拧盖轨迹冒充视频。
- C07 通过：观测 95/112，数值有限，10 个无动作控制步的转角、轴向位移和约束误差均为 0。

## C08：初始化问题，主动中止

- 启动 `--steps 327680 --evals 5`，输出 `artifacts/orca_cap/c08_ppo_scratch`。
- 编译期间检查发现 reset 的几何缓存未 forward，第一帧几何观测为零。
- 已中止，退出码 130；停止信号在 XLA 编译返回后生效。留下 step0 初始评测：成功率 0，
  平均最大角度 0，末尾约 -14.85 度，无已完成 PPO 更新。first/last/max 均仅指 step0，
  不能当作完整训练趋势。
- 下一轮类型：环境 reset，构建状态后显式 `mjx.forward`，再重跑同一探测。

## C09：修正 reset 后的纯 PPO 短探测

- 仅修正初始几何缓存，奖励和模型同 C08。随机网络 seed0，不加载历史策略。
- 命令：`python scripts/train_orca_cap.py --model artifacts/orca_cap/c06_time_constant/model.xml --output artifacts/orca_cap/c09_ppo_reset_fixed --steps 327680 --evals 5`。
- 学习率 3e-4，初始探索标准差 0.7，128 并行环境，4 个开发评测种子 1000--1003。
- 中途评测：81920 步为 3/4 成功，最大保持 9.25 秒；163840、245760 步均为 2/4。
  更大的转角并不代表更好，后两次有回合约束误差超过 1 毫米而被判无效。
- 完成 327680 步，625.33 秒（含启动/编译/评测）。当时原始计数（C15 修正前）
  success first=0，last=0.75，max=0.75；审计后 last=0.5。
  按成功率、平均有效保持、较早步数依次选择，best_step=81920，mean_hold=5.825 秒。
- 末尾平均保持 3.4375 秒，最大螺纹误差 2.228 毫米；非手指接触始终为 0。
  失败主因是部分回合约束误差，而不是简单“转不动”。不能隐藏末尾误差超限回合。
- 下一轮类型：训练初始化。只增加视频抓握轨迹的 BC，任务奖励、物理和 PPO 超参数不变；
  此时不调整奖励，以免混淆引导与奖励变化。

## C12：视频抓握 BC 初始化 + PPO，开发对照

- 从 seed0 随机网络开始，对 C11 的 4 条瓶盖环境回放进行 2000 次 BC，然后 PPO。
- 数据只有抓握先验，不包含成功拧盖示范；不加载 C09 权重，也不加载旧方块 PPO 权重。
- **不加持续示范损失**，否则会强迫策略继续复制仅转约 33 度的抓握动作。这与 F01 的
  BC 加持续模仿不同，必须明确标注，不能共用结论。
- 命令：`python scripts/train_orca_cap.py --model artifacts/orca_cap/c06_time_constant/model.xml --output artifacts/orca_cap/c12_video_bc_ppo --steps 327680 --evals 5 --bc-dataset artifacts/orca_cap/c11_video_grip_transfer/trajectories.npz`。
- 与 C09 使用同一 cap_v1 奖励、模型、reset、初始探索方差、PPO 参数、训练种子与评测种子。
  这里只做一对开发训练，不据此宣称统计显著的收敛优势。
- 已完成 327680 步，420.47 秒（含编译和评测），BC 本身 3.25 秒。
- BC 损失从 1.17193 降到 0.0002375；零步仍未拧盖成功，不能把模仿误差低当成任务成功。
- 按 C15 的整回合审计：success first=0，last=0.5，max=1.0，best_step=245760，
  平均有效保持 1.475 秒。末尾退化，不把最佳点冒充最终稳定水平。
- 下一步为独立复核选定与末尾模型。由于已经实际训练出符合拧松指标的策略，
  本轮不再改奖励追逐更高分，避免将开发调参混入本次对照。

## C13：纯 PPO 选定模型的独立原生复核

- 先依据 C09 的开发曲线选择 step81920，再使用未参与选择的种子 2000--2015。
- `evaluate_orca_cap_native.py` 使用原生 MuJoCo 重新运行策略闭环，不是回放保存的 qpos，
  接触力由 `mj_contactForce` 独立读取，转角/螺距从关节状态重算。
- 命令：`python scripts/evaluate_orca_cap_native.py --model artifacts/orca_cap/c09_ppo_reset_fixed/model.xml --policy artifacts/orca_cap/c09_ppo_reset_fixed/step_000081920/params.pkl --output artifacts/orca_cap/c13_scratch_native_heldout --count 16`。
- 16/16 通过 90 度并有效保持 1 秒；保持 4.55--6.75 秒，平均 4.6875 秒，末尾约 113 度。
- 最大螺纹误差 0.760 毫米，无非手指接触，最大法向接触力约 39.40 N。
- 两种视频使用首个新种子 2000 的同一轨迹：
  [碰撞版](../artifacts/orca_cap/scratch_native_videos/collision.mp4)、
  [STL 手外观版](../artifacts/orca_cap/scratch_native_videos/stl.mp4)。
- 这只验证小幅关节初始噪声的固定场景，不验证不同瓶子、位置、摩擦或真实硬件。

## C14：纯 PPO 选定模型的 MJX 新种子复核

- 与 C13 同样的 16 个新种子，重新加载同一选定权重，在训练后端再测一次。
- 命令：`python scripts/train_orca_cap.py --model artifacts/orca_cap/c09_ppo_reset_fixed/model.xml --output artifacts/orca_cap/c14_scratch_mjx_heldout --restore artifacts/orca_cap/c09_ppo_reset_fixed/step_000081920/params.pkl --eval-only --eval-start 2000 --eval-count 16`。
- MJX 与原生结果分别报告，不能用较好的那个覆盖较差的那个。
- 实测 MJX 10/16 成功，平均有效保持 4.125 秒，最大约束误差 1.918 毫米；无非手指接触。
  与原生 16/16 存在后端差异，不能宣称普适稳定。

## C15：整回合计数审计与修复

- 审计发现：原回合评测保留曾经达到的最长保持，即使之后出现超过 1 毫米的约束误差，
  仍可能被判成功。与预先写明的“整回合无违规”不一致，属于评测实现错误。
- 修复 `orca_cap_metrics.py`：任何时刻发生非手指接触、约束误差超限或数值/存活失败，
  整回合有效保持与成功均归零。没有修改物理或训练奖励，也没有重训以掩盖问题。
- 对两组所有 10 个开发检查点统一重算；旧 `metrics.csv` 与轨迹保留，新增
  `audited_metrics.csv` 和 `audited_episodes.json`。后续分析只读审计版。
- 受影响的成功率：C09 末尾从 3/4 改为 2/4；C12 step81920 从 2/4 改为 1/4，
  末尾从 3/4 改为 2/4。两个选定模型仍为 C09 step81920 和 C12 step245760，未改变。
- 审计后曲线：纯 PPO `[0, .75, .5, .5, .5]`；视频 BC + PPO `[0, .25, .25, 1, .5]`。
  两组步数均为 `[0, 81920, 163840, 245760, 327680]`。
- 测试共 6 项通过，新增覆盖“先成功后违规必须归零”和“接触中断必须重计时”。
- 原生 C13 没有任何回合误差超限，原结果不受影响；原生评测脚本也使用同一整回合审计。
- 这项修正不改变 F01 方块任务，拧盖与方块成功标准不同。

## C16：两组四个模型，统一 MJX 独立测试

- 依据审计后的开发曲线选定模型，不按独立测试结果重新选模型。
- 对纯 PPO、视频 BC + PPO 的选定模型及末尾模型，使用同一组新种子 2000--2015。
- 批处理重新加载权重，核对训练时保存的环境配置、模型字节和环境源码哈希。
- 命令：`python scripts/evaluate_orca_cap_pair.py --scratch artifacts/orca_cap/c09_ppo_reset_fixed --guided artifacts/orca_cap/c12_video_bc_ppo --output artifacts/orca_cap/c16_paired_heldout`。

| 模型 | C16 成功数 / 16 | 平均有效保持秒 | 最大螺纹误差毫米 |
| --- | ---: | ---: | ---: |
| 纯 PPO 选定 | 7 | 3.3219 | 5.254 |
| 纯 PPO 末尾 | 6 | 1.7625 | 3.853 |
| 视频 BC + PPO 选定 | 12 | 1.1938 | 2.161 |
| 视频 BC + PPO 末尾 | 10 | 1.6031 | 2.315 |

超过误差阈值的回合计失败，表内最大误差包括失败回合，不是成功回合允许这么大误差。
四个模型均未观察到非手指接触。C16 纯 PPO 选定模型与 C14 的 10/16 不同，因此进行
C18 原样重复，而不是选择更有利的评测入口。

## C17：引导选定模型的原生复核

- 与 C13 同样 16 个新种子，使用 C12 step245760，原生 MuJoCo 重新执行闭环策略。
- 13/16 成功，平均有效保持 2.0719 秒，无非手指接触，最大螺纹误差 1.358 毫米。
- 三个失败回合也会转到约 112 度，但约束误差违反整回合标准，所以仍判失败。
- 完整数据：`artifacts/orca_cap/c17_guided_native_heldout/`。

## C18：原样重复统一评测入口

- 使用相同脚本、四个权重、环境配置、16 个种子；不修改引导参数和物理。
- 输出：`artifacts/orca_cap/c18_repeat_paired_heldout/`。
- 成功数依次为纯 PPO 选定 9/16、末尾 9/16，引导选定 9/16、末尾 9/16。
- 与 C16 的 7/6/12/10 不一致，表明当前 GPU/MJX 运行存在可重复性风险。具体来源尚未
  在本轮定位，不能把原因肯定归给某一项，也不能平均两次就消除问题。
- 上游 [MuJoCo Playground 精度说明](https://github.com/google-deepmind/mujoco_playground#reproducibility--gpu-precision-issues)
  提醒默认矩阵乘法精度可能影响可重复性，并建议在实验启动前设置
  `JAX_DEFAULT_MATMUL_PRECISION=highest`。这是后续可检验的诊断方向，不是本轮已证实的原因。
- 本轮命令未显式固定这一精度设置，历史配置文件也未记录该字段。因此不应声称
  已完整控制所有底层数值条件。**不能只在测试时换精度，再把新结果混回原训练对照。**
  后续应从开发、训练到测试统一固定精度，再重跑成对实验并做重复性检查。

## 本轮结论与边界

1. 固定瓶身、Orca 单手、纯 PPO 与视频抓握 BC + PPO 的拧松实验均已实际跑通。
   两组都训练 327680 步，未给引导组更换奖励。模型和环境配置字节一致检查通过。
2. 整回合审计后开发曲线 AUC：纯 PPO 0.5000，引导 0.4375；**没有证据说这次引导更快**。
   一对训练种子、四个开发评测回合，以及重复测试波动，均不足以作算法性能排序。
3. 原生复核分别 16/16、13/16，但不能用原生的较好数据替换 MJX 负结果。
   这里只证明出现了有效的闭环拧松行为，不证明高可靠性和跨物体泛化。
4. 任务是至少 90 度且持续有效保持 1 秒，不是完整旋出取盖。螺纹是简化双关节约束，
   没有建模密封圈、真实螺牙、脱扣或取下后的自由运动。
5. 瓶身碰撞代理是长方体，瓶盖为 16 边凸网格；图中的绿色长方体表示固定瓶身代理，
   不是一个自由方块任务。STL 模式仅替换手的可视外观，不替换这些物理代理。
6. 瓶盖转轴为世界 +X，手和瓶身安装位置固定；扰动仅为初始手关节约 0.002 弧度噪声。
   关节摩擦参数、接触力及螺纹约束未做真实瓶子标定，不能直接部署硬件。

![全部审计后开发曲线](../artifacts/orca_cap/analysis/learning_curves.png)

## 训练和测试的一致性

| 项目 | C09 纯 PPO | C12 视频 BC + PPO | C16/C18 测试 |
| --- | --- | --- | --- |
| cap_v1 奖励、模型、动作接口 | 同一配置 | 同一配置 | 同一配置加载，仅评测不更新 |
| 17 维绝对角度，单步限幅 0.1 弧度 | 相同 | 相同 | 相同 |
| 控制 / 物理 dt | 0.05 / 0.002 秒 | 相同 | 相同 |
| PPO 学习率 | 3e-4 | 3e-4 | 不更新 |
| 初始探索标准差 | 0.7 | 0.7 | 确定性动作，两组一致 |
| BC | 无 | 2000 次，4 条抓握回放、960 样本 | 不再训练 |
| 持续示范损失 | 无 | 无 | 无 |
| 运行时基础视频动作 | 无 | 无 | 无，因此不是动作残差 |
| 初始随机化 | 手关节噪声 0.002 | 相同 | 相同分布，新种子 |
| 下层矩阵乘法精度 | 未显式固定/记录 | 未显式固定/记录 | 存在重复运行波动，尚需进一步诊断 |

训练网络和物理参数没有在测试时人为调成更有利的值，但没有充分固定数值精度是本轮
控制变量的缺口。源代码和配置一致检查不能替代实际重复测试。

## 视频与策略位置

以下来自 C16 的同一个新种子 2000；每段 12 秒、20 FPS、960x720，碰撞与 STL 两版
使用相同保存状态，画面标注真实转角及螺纹误差。不是重新仿真两次制作两种外观。

- 纯 PPO：[碰撞几何版](../artifacts/orca_cap/scratch_mjx_videos/collision.mp4)、
  [STL 手外观版](../artifacts/orca_cap/scratch_mjx_videos/stl.mp4)，该回合有效保持 11 秒。
- 视频 BC + PPO：[碰撞几何版](../artifacts/orca_cap/guided_mjx_videos/collision.mp4)、
  [STL 手外观版](../artifacts/orca_cap/guided_mjx_videos/stl.mp4)，该回合有效保持 1.25 秒。
- 纯 PPO 原生复核视频见 C13，与上述 MJX 视频明确分开。

选定权重：

```text
artifacts/orca_cap/c09_ppo_reset_fixed/step_000081920/params.pkl
artifacts/orca_cap/c12_video_bc_ppo/step_000245760/params.pkl
```

这些是仿真策略，不是硬件固件；回放依赖当前工程的 Orca 模型资产和 `aero_rl` 环境。
在主工程中运行本日志中的命令时，需要先设置：

```bash
export PYTHONPATH="$PWD/Aero-Hand/sim_rl/mujoco_playground"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/ll/miniconda3/envs/aero_rl/bin/python
```

上文 `python` 可替换为 `"$PY"`；视频渲染另需 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`。
输出目录要求不存在，防止覆盖旧实验。所有中止、失败和修正前结果保留。

交付复核：6 项自动测试通过；`verify_orca_cap_delivery.py` 校验了开发与独立测试的
逐帧转角、螺距误差、成功计数、策略/轨迹哈希及视频来源。记录位于
`artifacts/orca_cap/delivery_audit.json`。方块 4 段、拧盖 MJX 4 段及原生 2 段视频均通过
完整解码检查；两种手外观的截图已人工查看，所有训练、评测和渲染进程均已结束。

## 下一阶段优先顺序

- 先解决数值可重复性与螺纹约束误差，在训练测试全程固定精度，做无动作、给定控制、
  重复闭环三层检查。不能先继续改奖励，再把求解变化当作引导效果。
- 收集真正含切向旋转和重新抓握的视频，按瓶盖轴建立物体局部坐标映射。
  现有方块视频教的是夹住物体，不能提供完整开盖技能。
- 用同一新任务版本做纯 PPO、BC 初始化、持续模仿、动作残差四种消融，分离各组件作用。
- 如需完整取盖，增加脱离螺纹的机制、轴向抬离空间与任务判据；不能把当前 90 度子任务
  改名为“完整开瓶盖”。

## C10：相同控制目标，原生与 MJX 对照

- 命令：`python scripts/replay_orca_cap_controls.py --model artifacts/orca_cap/c06_time_constant/model.xml --reference artifacts/orca_cap/c06_time_constant/ik_controls.npz --kind procedural_ik --output artifacts/orca_cap/c10_mjx_reference_check --count 1 --noise 0`。
- 不训练策略，关闭初始噪声，回放与 C06 完全相同的控制序列。
- MJX 最大 74.966 度，末尾 65.743 度；原生分别为 72.575、62.475 度。
- MJX 最大螺纹误差 0.499 毫米，原生为 0.327 毫米；两者均无非手指接触。
- 两者定性一致但非逐位相同；仍未达 90 度。成功统计必须保留误差检查，不能只看转角。
- 三项原生自动测试通过：瓶身固定/手电机不驱动盖子、零动作负对照、螺距几何关系。
- 原生程序诊断视频（不是 PPO）：`artifacts/orca_cap/c06_native_visuals/{collision,stl}.mp4`。

## C11：仅迁移视频抓握先验，检查覆盖范围

- 源为已有 HIC/DexMV 的 Orca 控制轨迹 `artifacts/orca30/dexmv_slow_acquisition_o21/reference.npz`。
- 命令：`python scripts/replay_orca_cap_controls.py --model artifacts/orca_cap/c06_time_constant/model.xml --reference artifacts/orca30/dexmv_slow_acquisition_o21/reference.npz --kind video_grip_transfer --output artifacts/orca_cap/c11_video_grip_transfer --count 4`。
- 不生成虚构的视频拧盖动作，不加载成功的方块 PPO 权重；这是跨任务先验直接回放。
- 四个回合全部未达到 90 度。平均最大 33.341 度，末尾 33.085 度，配对接触 89.17%，
  无非手指接触；最大约束误差 0.736 毫米，峰值接触力 15.03 N。
- 说明抓握先验能接触瓶盖，但不会自动变成拧盖策略。数据保留，后续可以仅 BC 初始化，
  再让 PPO 学额外的转动；不持续强迫网络复制没有拧盖动作的旧参考。
