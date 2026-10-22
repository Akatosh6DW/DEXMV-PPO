# 圆柱 S07：样本贡献与残差控制复验

这是 S06 三指指腹抓持的扩展，不覆盖之前的方块 S05、圆柱 S02 或 S06。

**已完成：9次正式训练、86项评估/消融、3424个测试回合。** 先看 [完整结果说明](结果说明.md)，里面从样本来源、训练流程和术语讲起，给出全部主表、图、参数和结论；原始数据入口见 [数据目录与复查方法](数据目录与复查方法.md)。

正常条件：仅BC70.31%、残差67.71%、参考59.38%、BC+PPO1.56%、纯PPO0%。残差不是全面第一，也不是永不退化；低摩擦和加重条件几乎都失败。报告保留这些边界，不把开发集100%当作独立成功率。

先读 [样本意义与残差参数](样本意义与残差参数.md)：用人话说明“抓得不像是否意味着视频没用”、残差公式、每个参数的含义，以及本项目视频先验的实际边界。

接触判据：[指腹和指尖如何判定](指腹和指尖如何判定.md)。解释圆柱三指抓持、局部坐标、接触方向和力，以及为什么外观上拿住不一定算严格成功。

再读 [奖励函数与样本来源](奖励函数与样本来源.md)：列出真正生效的奖励公式、各项门槛、离线校正与在线PPO的区别，以及从发布包到11条BC回合的完整来源链。

专项分析：[BC+PPO为何退化：理论与证据](BC加PPO为何退化_理论与证据.md)。包含实际动作漂移、训练日志中的KL、探索尺度、价值初始化、分布偏移等证据与待验证假设。

三者对照：[残差为何更能保留抓法](残差为何更能保留抓法_与BC对照.md)。一起解释仅BC、BC+PPO和残差的区别；残差也有回落，不宣称永不退化。

架构澄清：[残差架构与BC输出的关系](残差架构与BC输出的关系.md)。当前是参考轨迹作基准，不是BC作基准；也解释怎样才算“BC基准＋PPO修正”。

跨任务解释：[为什么方块改善而圆柱退化](为什么方块改善而圆柱退化.md)。核对真实配置，并给出一个重要反例：方块也偏离BC，但任务成功率提高；相似度不等同于任务性能。

后续任务：[S08圆柱掌面包覆](../S08/README.md)。先完成几何参考与摆位修复，再完成三方法各3种子的完整比较；不改S07的三指指腹标准，不能跨任务直接排名。S08使用几何参考，不是新RGB视频提取。

固定方案见 [实验约定](实验约定.md)，真实过程见 [迭代日志](iteration_log.md)。输入、权重及源码快照哈希、初态配对、1604条已保存轨迹的计时/成功复算通过。27项回归测试通过；正常条件和消融保存全轨迹，扰动条件每批保存前4条，不声称全部3424条都有完整轨迹复算。

手心短视频：[S07目录](../../../deliverables/S07_圆柱完整复验_2026-10-12/目录.md)；新的包覆方向：[S08目录](../../../deliverables/S08_圆柱包覆可行性_2026-10-12/目录.md)。均包含外观/碰撞几何以及失败案例，文件名区分任务和方法。

## 文件位置

- 原始实验：`artifacts/orca_benchmark_suite/S07/`。
- `inputs/`：固定圆柱模型与校正参考。
- `inputs_uncorrected/`：同一物理场景，校正前参考。
- `inputs_linear/`：同一初态与终点，线性闭合参考，仍包含视频终态先验。
- `train/`：三个方法、各三个种子的完整开发指标和权重快照。
- `selection.json`：全部训练完成后、独立测试前冻结的选择。
- `eval/`：各条件逐回合结果和初始状态；正常条件保存全部轨迹，扰动条件保存前四条示例轨迹，所有回合指标均保留。
- `report/`：最终聚合 CSV、图表与核验结果。

## 关键源码

| 文件 | 做什么 |
|---|---|
| [retarget_s06_cylinder.py](../../../scripts/retarget_s06_cylinder.py) | 物体相对位置、末端方向约束和跨手型 IK |
| [refine_s06_reference.py](../../../scripts/refine_s06_reference.py) | 离线有界仿真校正；不是 PPO |
| [orca_s06_env.py](../../../scripts/orca_s06_env.py) | 判断三指末节指腹是否真正承力 |
| [orca_s07_env.py](../../../scripts/orca_s07_env.py) | 保持正常条件不变，正确施加质量/摩擦等扰动 |
| [orca_residual_controls.py](../../../scripts/orca_residual_controls.py) | 参考重采样及基准＋修正的计算 |
| [orca_residual_env.py](../../../scripts/orca_residual_env.py) | 执行残差、裁剪、记录请求与实际修正 |
| [run_s05_job.py](../../../scripts/run_s05_job.py) | 共用的 BC、PPO 训练和逐回合评估实现 |
| [run_s07_job.py](../../../scripts/run_s07_job.py) | S07 参数入口，历史版本不受影响 |
| [run_s07_study.py](../../../scripts/run_s07_study.py) | 固定预算执行、日志、权重冻结、配对测试 |
| [analyze_s07_study.py](../../../scripts/analyze_s07_study.py) | 原始数据聚合、保持时间复算、相似性、图表 |

## 复现命令

环境为 `conda activate aero_rl`。GPU 训练需可见的 NVIDIA 设备，原生 MuJoCo 做物理，JAX/Brax 做策略计算。以下命令仅在对应输出不存在时实际执行；已完成任务会校验并跳过，未完成任务不会自动覆盖。

```bash
python scripts/run_s07_study.py prepare
python scripts/run_s07_study.py train
python scripts/run_s07_study.py evaluate --workers 3
python scripts/analyze_s07_study.py
python scripts/write_s07_report.py
```

统一设置 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`。精确启动命令在 `train/*.log` 和 `eval/*/*.log` 第一行，运行参数与源码哈希在各任务目录 `run_config.json` 中。S07 附加参数记录在 `env_config.json` 的 `s07` 字段；不要只看公共 CLI 中未使用的默认项。
