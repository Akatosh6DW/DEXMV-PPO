# 运行、复算与查看方法

[返回目录](README.md)。命令均在主工程目录执行，不在Aero-Hand或DexMV子目录执行。

## 环境

```bash
cd /home/ll/SRTP/DEXMV-PPO
conda activate aero_rl
```

本轮训练与评测使用`aero_rl`里的JAX/Brax与MuJoCo。视频跟踪、重定向和参考校正结果已经离线保存，
训练时不必同时运行DexMV环境，也没有把DexMV的TRPO策略直接当作PPO权重加载。
模型、参考、示范数据与源码哈希保存在S03的协议JSON中；匹配的是数据与动作接口。

## 先检查是否仍在运行

```bash
systemctl --user status orca-s03-confirmation --no-pager
journalctl --user -u orca-s03-confirmation -n 20 --no-pager
tail -n 20 artifacts/orca_benchmark_suite/S03/confirmation_execution.jsonl
```

`ActiveState=active`且MainPID仍存在时，观察原服务，不要再次启动同一批任务。
`Result=success`单独出现不代表全实验完成；服务仍active时该字段也可能显示success。
真实完成需结合逐作业`summary.json`、最终矩阵标记、统计、视频与审计判断。

开发阶段服务为`orca-s03-development`，已完成24项候选训练。
确认服务先训练40个模型，再标定4个外力候选、封存最终评测矩阵、完成评测与导出。
检查点保存期间看到新文件不等于作业已完成；以该作业`summary.json`的completed状态为准。

可在队列运行时只读核验已完成训练（不会重新训练或访问正式测试）：

```bash
python scripts/audit_orca_s03_training.py
```

它更新`training_audit.json`，明确列出已核验和待完成数量；不是最终实验审计。

## 文件对应关系

所有数据在`artifacts/orca_benchmark_suite/S03/`：

| 路径 | 用途 |
|---|---|
| `development_protocol.json` | 开发预算、选优规则、冻结输入 |
| `development/<方法>/r<轮次>_s<种子>/` | 每个候选的完整日志与权重 |
| `development_rounds/` | 每轮保留/拒绝决策及分数 |
| `selected_recipes.json` | 开发结束时封存的配方；其中confirmation_not_started表示封存当时状态，不是实时进度 |
| `confirmation_protocol.json` | 十个新训练种子、预算、主比较、视频选择规则 |
| `confirmation/<组>/s<种子>/` | 长训练、BC初始化（适用时）、9个检查点 |
| `force_development/`、`force_selection.json` | 仅用开发初态确定外力，不按残差优势挑外力 |
| `evaluation_protocol.json` | 正式测试开始前封存的738项作业 |
| `eval/<条件>/<组>/.../evaluation/` | 初态、逐回合指标、前4条完整轨迹 |
| `mechanism/s<种子>/` | 四条闭环/固定回放执行路径与无扰动动作表 |
| `report/` | 完整统计CSV、结论表、学习曲线与图表 |
| `videos/index.md` | 固定案例视频索引 |
| `audit.json` | 输入、预算、配对初态、评分与视频的核验范围 |

执行中尚未到达的阶段不会提前生成最终产物。不存在的文件不表示已经做过但被隐藏。
文档与原始数据各自只保留一份，通过相对链接关联，不复制权重和视频到docs。

## 如何看权重与动作

例如完整残差长预算权重在完成后位于：

```text
artifacts/orca_benchmark_suite/S03/confirmation/residual_tuned/s21001/step_000655360/params.pkl
```

163840步目录是预定短预算；655360步是预定长预算，不按测试结果择优。
`bc_params.pkl`是BC后、PPO前的初始化；`params.pkl`为该训练作业最后保存的参数。
这些是本项目Brax/JAX网络参数，不是可直接给任意手型或任意`view.py`加载的通用模型。
需要配套的手型、观测顺序、动作缩放和环境配置，评测脚本会核对来源任务和种子。

普通视频只显示保存的状态，不能仅凭画面推断策略是否有反馈。
分析`video_base`、`requested_residual`、`effective_residual`、`ctrl`字段，才能区分
参考动作、请求修正、限幅后实际修正和最终关节目标。
机制实验还保存`nominal_recorded_actions.npz`以及`feedback_diagnostic.json`。

## 测试与完整复算

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true python -m unittest discover -s scripts -p 'test_orca_s03*.py'
```

以下命令在正式矩阵完成后执行；缺失作业会报错，不会把缺失填作零成功率：

```bash
python scripts/report_orca_s03.py
python scripts/plot_orca_s03.py
python scripts/render_orca_s03.py
python scripts/audit_orca_s03.py --require-videos
python scripts/supplement_orca_s03_quality.py
```

报告允许`--allow-partial`输出明确标注的中间统计，不能用它证明全矩阵完成。
视频脚本会验证已完成文件的身份并跳过，不自动覆盖未完成的目录。
独立评分复算覆盖每个作业保存的前4条轨迹；全部回合保留子步审计汇总，
不能把“前4条独立复算”写成“全部15万回合独立重放过物理”。
质量敏感性补充单独生成`quality_sensitivity.md`及CSV，不改变主报告和主假设检验。

## 中断后的处理

先查服务状态、实际进程及对应日志，确认是进程结束，而非观察超时。
已完成且身份匹配的作业可以跳过；没有summary的已有目录视作部分尝试，运行器会拒绝覆盖。
保留失败日志，查明原因后才安排单独重试并记录关系，不删除旧目录伪装从未失败。
不要修改封存配置、数据或源码来让运行器通过哈希检查。
只有确认原服务已退出且没有部分作业需要处理时，才可运行：

```bash
python scripts/run_orca_s03_confirmation.py --workers 3
```

该命令继续固定矩阵，不是新一轮调参。修复确实需要改变冻结实现时，应创建明确的版本修订，
保留旧协议和失败尝试，而不是悄悄把前后结果混成同一版本。
