# DEXMV-PPO：视频先验与灵巧手强化学习

本工程研究怎样把人类操作示范转成机械手可执行的抓法，再用PPO学习。主线是Orca机械手抓方块、抓圆柱；Aero-Hand保留原PPO任务和模型展示入口，DexMV提供示范处理与重定向相关工具。

工程同时实现两种融合：**BC后PPO**先模仿再强化学习；**参考加残差PPO**在执行每一步时保留参考动作，由PPO提供有界修正。实验还包含不使用示范初始化的PPO，以及不加修正的参考回放。

## 当前成果

截至2026-10-28（实验数据截至2026-10-24）：

- **S10整手对准圆柱**已经完成三训练种子对比、共同初态测试、逐例统计和手心视频。先用物体位姿把整手对准，再执行手指抓法。原成功标准下，BC后PPO为93.49%，参考回放和残差PPO均为91.41%。
- **S11固定瓶身开盖**完成小规模机构与接触测试。校正接触后能拧到脱扣，但取盖时失去夹持，未完成取下并拿稳。暂不开展大规模开盖策略训练。
- **S12软圆柱抓取**已完成三训练种子、65组共1560回合独立评估与十段手心视频。标准条件严格成功率：BC起点95.83%、BC后PPO80.21%、参考50%、残差PPO44.79%、纯PPO0%；没有证明PPO稳定优于起点。21回合局部单元失效已计失败，不能声称所有物理轨迹有效。[完整结果与资料入口](docs/experiments/S12/README.md)。

先读[S10通俗讲解](docs/experiments/S10/从头理解整手对准实验.md)、[S10结论](docs/experiments/S10/结论与分析.md)和[S11论文调研及可行性记录](docs/experiments/S11/README.md)。

## 方法原理

```text
源视频 / 已处理的人类操作示范
             |
       手与物体的位姿序列
             |
   跨手型重定向 + 接触几何校正
             |
      物理回放与成功示范筛选
             |
    +--------+---------------------+
    |                              |
  BC训练                         参考轨迹
    |                              |
  PPO继续更新                  每步 + PPO修正量
    |                              |
 完整动作网络                    残差控制器
```

**BC（行为克隆）**：用“当时看到了什么、示范做了什么”训练网络。BC后PPO继续调整这个网络；最终运行时通常不再把示范动作显式相加。

**PPO（近端策略优化）**：策略与仿真交互，环境给出奖励，算法用收集到的状态、动作和回报更新网络。正式对比期间固定奖励和训练配方；早期人工/大模型读日志改奖励的开发迭代另行记录。

**残差PPO**：每个控制周期计算“关节目标 = 参考关节目标 + 有界PPO修正”，再执行关节限位和限速。参考轨迹在运行时始终参与控制，不只是训练前的起手提示。

**重定向**：把人手的指尖、指腹或关节运动映射到机械手。由于两种手的尺寸和自由度不同，需要匹配几何目标、限制关节范围并检查碰撞。校正后轨迹不等于未经处理的视频原姿态。

**整手对准（S10）**：保持示范中“物体到手”的相对位姿，按当前物体位姿计算新手位姿。整手负责接近和朝向，手指网络负责闭合与调整。当前用理想笛卡尔平台驱动自由基座，不是已验证的实机机械臂。

## 实验索引

| 实验 | 要回答的问题 | 文档 |
|---|---|---|
| 方块S02 | 视频示范引导与残差在固定预算、不同扰动下如何表现？ | [设计、数据与结果](docs/experiments/S02/README.md) |
| 方块S04 | 有限在线交互预算下，各方法的表现如何？ | [基准设计](docs/experiments/S04/README.md) |
| 圆柱S02 | 把方块任务换成圆柱，重新准备示范并比较同样方法 | [执行记录](docs/experiments/cylinder_s02_execution.md)、[通俗解读](docs/experiments/cylinder_s02_results_explained.md) |
| S05至S08 | 指腹接触、三指抓持、复验及纵向包覆抓法 | [S05](docs/experiments/S05/)、[S06](docs/experiments/S06/)、[S07](docs/experiments/S07/)、[S08](docs/experiments/S08/) |
| S09 | 固定基座适应不同圆柱朝向，以及BC后PPO退化的修复 | [总入口](docs/experiments/S09/README.md) |
| S10 | 整手先对准，能否共享同一相对抓法？ | [完整协议和结果](docs/experiments/S10/README.md) |
| S11 | 固定瓶身，Orca单手能否旋拧并取下瓶盖？ | [小规模可行性](docs/experiments/S11/README.md) |
| S12 | 视频几何先验换成可变形材料后，能否稳定抓持并减少挤压？ | [完整结果、原理与审计](docs/experiments/S12/README.md) |

S10的1,792条策略测试记录使用128个共同初态，另有512条几何对照记录。主要策略各训练3次，每次163,840次在线交互；BC的20,000次监督更新单独计算。参考回放本身已经较强，本轮没有观察到残差相对参考的稳定净收益。完整成功要求撤去支撑后连续合格抓持30秒、结束时仍合格，并通过接触质量检查。

每个任务的初态、对象尺寸、支撑撤除时刻、成功判据和示范来源以对应协议为准。S10是抓住初始悬挂的圆柱，不是从桌面拾起；不同实验表格对应不同任务条件。

## 工程布局

| 目录 | 内容 |
|---|---|
| `Aero-Hand/` | 自有分支子模块；PPO环境、Aero/Orca模型、展示脚本 |
| `DexMV/` | 自有分支子模块；示范仿真与重定向工具 |
| `DexMV-Learn/` | 上游学习代码子模块；不向原作者仓库推送 |
| `scripts/` | 示范处理、几何校正、训练、评估、绘图、导出和归档 |
| `docs/experiments/` | 按实验编号集中管理的方案、日志和结论 |
| `docs/literature/` | 文献综述与相关材料 |
| `artifacts/` | 本机原始运行输出：权重、轨迹、逐例指标、视频 |
| `deliverables/` | 本机汇报资料和专题归档；不再重复生成大ZIP |
| `release/` | 本次发布的权重、参考、便携场景、冻结源码和评估数据 |

`artifacts/`与`deliverables/`是工作区数据，不把所有历史中间产物加入Git。[发布目录](release/README.md)将245份去重参数与1,141项评估绑定，包含7套便携场景和逐版本源码；最新S10策略可直接使用发布入口运行。

## 环境与启动

**新开发者请先读 [安装与续训入口](docs/development/README.md)**。本次发布增加了 S12 训练样本、最终权重、统计与逐帧审计附件；旧实验策略见 [POLICIES.csv](release/POLICIES.csv)。两人协作约定见 [CONTRIBUTING.md](CONTRIBUTING.md)，可复用提示词见 [项目提示词](docs/development/prompts.md)，上传/排除范围见 [发布清单说明](docs/development/publication.md)。

本机PPO及模型展示环境名为`aero_rl`：

```bash
conda activate aero_rl
python Aero-Hand/view.py
```

仅克隆发布内容、没有本机历史数据时，展示脚本会自动读取`release/`里的对应场景和参考。运行发布的S10策略：

```bash
python scripts/replay_release_s10.py --case guided_s22101 --output outputs/release_guided
```

展示脚本的默认模型和快捷键说明保留在`Aero-Hand/view.py`中。展示模型不等于运行训练好的策略；策略评估需要匹配的任务场景、观测定义、参考和权重。

PPO端以MuJoCo、JAX/Brax和本地`mujoco_playground`为主。DexMV旧环境与PPO环境分离，通过`.npz`交换轨迹；两边不需要同时安装在同一个Python环境中。环境与接口细节见[接口梳理](docs/dexmv_aero_interface_survey.md)和[桥接流程](docs/dexmv_to_residual_ppo_pipeline.md)。

已配置机器上的接口测试：

```bash
python -m unittest discover -s scripts -p 'test_s10*.py'
python -m unittest discover -s scripts -p 'test_s11*.py'
python -m unittest discover -s scripts -p 'test_s12*.py'
```

完整数据仍在本机时，可运行`python scripts/verify_s10_delivery.py`核对S10权重来源、预算、共同初态与视频。该命令需要完整实验输出，不是仅下载源码后的安装测试。

## 克隆与开发

```bash
git clone --recurse-submodules https://github.com/Akatosh6DW/DEXMV-PPO.git
cd DEXMV-PPO
# 已经克隆主仓库但未下载子工程时：
git submodule update --init --recursive
```

子模块固定在主仓库记录的提交，不会自动跟随原作者更新。修改子工程后，先在自己的子仓库提交、推送，再在主仓库提交新的子模块指针。Aero-Hand和DexMV的`origin`均指向Akatosh6DW名下仓库。

## 汇报、数据与许可

方块两轮合并讲解：[实验总报告和中文注释源码](docs/experiments/Orca_Video_PPO/README.md)。本机完整资料入口为`deliverables/完整实验汇报资料_2026-10-12/`；S10视频在`artifacts/orca_benchmark_suite/S10/videos/`，S11诊断视频位置见对应文档。

模型、第三方代码和视频示范分别遵循原许可。DexMV、HIC、OakInk等来源应在使用对应数据时署名并核对其许可；公开项目源码不代表源视频或派生数据可以任意商用。原始视频不随源码仓库整体上传。

[发布前清理记录](docs/experiments/maintenance/2026-10-23_release_cleanup.md)记录了旧压缩备份和早期训练检查点的处理依据与保留规则。
