# 开发者接手入口

本发布针对 Linux x86_64、Python 3.11。主线是 Orca 方块、刚性圆柱、S12 软圆柱。不是硬件控制系统；相机感知、实机安全控制和机械臂动力学没有因此完成。

## 1. 下载与安装

```bash
git clone --recurse-submodules https://github.com/Akatosh6DW/DEXMV-PPO.git
cd DEXMV-PPO
conda env create -f environment/aero_rl.yml
conda activate aero_rl
bash scripts/bootstrap_developer.sh
```

若已有 `aero_rl`，不要覆盖环境，先比较版本；也可使用 Python 3.11 的独立 venv。CPU 依赖及传递依赖分别锁定在 `environment/requirements-aero.txt`、`environment/constraints-aero-linux-py311.txt`。首次安装需要网络，不是零依赖免安装程序。

有 NVIDIA GPU 时，在上述环境中安装对应扩展，然后检查设备：

```bash
python -m pip install 'jax[cuda12]==0.9.0.1' -c environment/constraints-aero-linux-py311.txt
python -c 'import jax; print(jax.devices())'
```

GPU 运行需要宿主机可用 NVIDIA 驱动；原实验机器为 RTX 4060 Laptop 8 GB。CUDA 扩展不是 CPU 环境的必需项。软体接触物理仍在原生 MuJoCo CPU 工作进程执行，GPU 主要负责网络与 JAX 计算。无 GPU 可用 `JAX_PLATFORMS=cpu` 做接口验证。图形窗口需桌面显示服务；无头渲染需要 EGL/OSMesa 及系统图形库，视频编码需要 ffmpeg。

## 2. 继续最新策略

先用一个开发初态检查加载，再继续训练：

```bash
python scripts/continue_s12.py --case guided_s25211 --eval-count 1 --output outputs/alice/check_load
python scripts/continue_s12.py --case guided_s25211 --steps 51200 --output outputs/alice/guided_continue
python scripts/continue_s12.py --case residual_s25211 --steps 51200 --output outputs/bob/residual_continue
```

`ppo` 是从零训练过的 PPO；`guided` 是 BC 后 PPO；`residual` 是每步参考动作加 PPO 修正。训练种子可选 25211、25212、25213。`--steps` 是**新增**在线交互数，必须为 640 的倍数；默认 0 只评估。输出目录必须是新的。

**这是权重热启动，不是完整断点恢复**：恢复 actor、critic 和已有归一化参数，但 Adam 优化器、随机数和仿真状态重新初始化。旧权重没有保存这些状态，无法凭空恢复。51200 步之后再跑 51200 步，是新的继续训练实验，不是重现原来的独立训练种子；不要把结果偷偷写回旧表格。

从相同初始配方重训，而非读取最终权重：

```bash
python scripts/continue_s12.py --case guided_s25211 --from-scratch --steps 51200 --evals 6 --output outputs/alice/guided_fresh
```

guided 会先用配套样本做 10000 次 BC 更新。另两组将方法改为 ppo/residual；参考回放没有神经网络权重。此入口保持 S12 已冻结的算法不变，仅适配迁移后的资产路径；若要改奖励、观测、网络或任务，请建立新实验编号及协议，勿修改旧结果的身份。

## 3. 数据和旧策略在哪里

- [策略总索引](../../release/POLICIES.csv)：旧实验所有实际评估参数及 S12 最终参数，按实验、方法、种子索引；`storage` 相对 `release/`。
- [原发布清单](../../release/manifest.json)：S02/S04、圆柱 S02、S05–S10、BC 修复与消融实验，245 个唯一参数载荷、1141 项评估。
- [扩展清单](../../release/developer_manifest.json)：S12 最终与开发记录、学习样本、逐帧轨迹、视频、SHA-256、来源路径和附件位置。
- [S12 正式统计表](../../release/records/artifacts/orca_benchmark_suite/S12/study_v2/report/aggregate.csv)、[人话报告](../experiments/S12/实验总结.md)。

权重不是单独通用模型，必须配套观测定义、动作语义、参考和场景。历史 `run_config.json` 中的旧绝对路径是来源记录，不是新机器启动命令。S10 发布推理入口仍是 `scripts/replay_release_s10.py`；S02/S04 等旧权重及源码已完整保留，但不声称所有历史训练器都拥有本次新增的 S12 一键续训适配。

```bash
# 恢复旧实验的配置、权重、样本及报告到对应工作路径。
python scripts/restore_developer_release.py --all
# 需要逐帧物理审计时再下载 Release 附件，日常训练不需要。
python scripts/restore_developer_release.py --all --trajectories
# 只核对已有发布文件，不写工作目录。
python scripts/restore_developer_release.py --all --verify-only
```

恢复器优先使用相对符号链接，避免再拷贝重复权重；XML 仅迁移资产路径，不改变几何数值。不会覆盖与发布内容不一致的本地文件。Linux 是当前支持目标，Windows 请用 WSL2/Linux 文件系统，不要依赖不支持符号链接的共享盘。

## 4. 验证与分析

```bash
python -m unittest discover -s scripts -p 'test_developer_release.py'
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu python -m unittest discover -s scripts -p 'test_s12_*.py'
python scripts/verify_policy_release.py --release release
```

S12 共 65 个独立评估作业、1560 回合。统计表不能把同一参考回放伪造为 3 个训练种子；标准条件 BC 起点 95.83%、BC 后 PPO 80.21%、参考 50%、残差 44.79%、纯 PPO 0%。21 回合发生单元失效，计失败而非删去。奖励提高不等于抓取质量提高。详见 S12 报告及配对置信区间。

原冻结协议与评估计划保留了原机器路径；跨机器重跑批量调度必须生成新输出协议，不能直接执行其中的旧 argv。本次保证的便携执行入口是上述 S12 训练/评估入口及既有 S10 推理入口，原始统计文件可直接读取，无需运行历史调度器。

## 5. DexMV 和原始视频的边界

继续 PPO 不需启动旧 DexMV 环境：发布中已有选定的人类动作数值样本、重定向参考、物理成功示范和 BC 数据。S12 用的是处理后的 DexMV seq96 与 S09 模板 0，并非本轮重新从 RGB 视频训练感知模型。

若重新处理源视频，请另建 `dexmv` 环境，参考 `DexMV/environment.yml` 与[原训练说明](../dexmv_policy_training_guide.md)。原工作环境 Python 3.7.11、NumPy 1.20.3、SciPy 1.7.1、Torch 1.8.1、gym 0.19.0、mujoco-py **2.1.2.14**、Cython **0.29.37**，使用 MuJoCo 2.1；上游 yml 中 mujoco-py 2.0.2.13 是旧配方，不要当作本机精确锁。源视频/完整数据集、MuJoCo 旧二进制、个人 Python 环境不随仓库上传，需按来源许可自行下载。这条旧感知/重定向链没有在此次新建环境中重装验证。

两人工作约定见根目录 [CONTRIBUTING.md](../../CONTRIBUTING.md)，可复用项目提示词见 [prompts.md](prompts.md)，上传范围见 [publication.md](publication.md)。
