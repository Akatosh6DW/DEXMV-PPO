# 实验策略与数据发布目录

2026-10-28 扩展：[开发者入口](../docs/development/README.md)、[策略总索引](POLICIES.csv)、[扩展清单](developer_manifest.json)。保留本页描述的旧发布；新增 S12 全部正式结果、最终/阶段参数、训练样本、开发诊断、视频，以及 Release 中去重逐帧轨迹。重复文件改为相对链接，旧清单的 SHA 验证仍有效。下面的 245 份参数和体积描述仅指旧发布，不包含本次增量。

本目录与主工程一起版本管理，包含方块S02/S04、圆柱S02、S05至S10及S09后续确认实验实际评估使用的权重、配置、源码快照、参考和场景。不是按某次最高分挑一个模型，而是按评估记录回溯所用检查点。

## 内容

- `manifest.json`：1,141项评估与245份去重参数文件的对应关系，注明方法、输入、SHA-256和原始位置。
- `blobs/`：按内容哈希存储的权重、参考、原配置输入与冻结源码。相同文件只保存一次。
- `scenes/`：7套场景，每套含相对路径XML、资产和同版本MuJoCo编译模型MJB。
- `records/`：原训练/评估配置、指标、任务协议及汇总表；S11是失败可行性记录，没有成功开盖网络。
- `build.json`：打包时的文件数、大小和主清单哈希。

245份参数包含正式方法、消融、BC起点、后续确认实验及严格零修正诊断；不能把它们都当作245个独立训练种子。参数文件合并去重后约71MiB，清单覆盖的文件内容约205MiB；加上清单及文件系统分配后本机占用约233MiB。不含原始视频和大规模训练轨迹。

## 找到指定策略

用主清单中的`inventory.evaluations`按`group`和`method`筛选，再通过`policy_sha256`查询`blobs`。例如S10的`guided`是BC后PPO，`residual`是残差PPO，`ppo`是从零手指PPO。训练种子与对应来源保存在该条目的`config`文件中，精确副本可在`records`映射下找到。

`policy_sha256`为空通常表示直接回放参考，不需要网络；S10严格零修正诊断保留了零均值未训练网络，条目中另有`policy_role`。

不同方法的动作含义、归一化参数和网络结构不同。不要只取一个PKL交给任意环境；应配套使用其冻结源码、场景和参考。PKL只应从可信来源加载。

## 核验

在已配置的主工程中：

```bash
conda activate aero_rl
python scripts/verify_policy_release.py --release release
```

该命令检查文件哈希、参数可加载且为有限值、参考数组与模型初始化。它不重新运行全部性能测试。

XML使用相对路径，便于迁移和查看。凸网格重新编译时可能出现多边形起点顺序或约1e-11量级的法向差异；打包时逐项检查了循环顶点等价及法向误差。MJB由原XML编译并逐数组核验，使用MuJoCo 3.5.0加载，适合需要保持本轮编译模型的复验。

## 运行S10策略

```bash
python scripts/replay_release_s10.py --case guided_s22101 --output outputs/release_guided
python scripts/replay_release_s10.py --case ppo_s22101 --output outputs/release_ppo
python scripts/replay_release_s10.py --case residual_s22101 --output outputs/release_residual
```

入口从发布清单恢复该案例的冻结源码和参考，解析便携XML资产，加载对应权重并运行42秒仿真，输出实际控制轨迹与保持时长。可以用`--seed`改变测试初态。输出目录必须尚未存在。

评估记录里的旧绝对路径保留为来源信息，不是新机器的启动配置。上述入口不读取旧`artifacts/`输入，并主动拒绝Python层对历史实验目录的读取；源网格也逐文件绑定到发布目录。单个CPU案例用于验证可迁移运行，不重新宣称GPU批量测试的统计结果。

## 数据来源

参考序列来自对应实验记录的人类示范、跨手型重定向和接触校正。原视频、HIC/OakInk/DexMV数据与派生内容仍遵循各自许可，不能从本目录公开推导为原数据可自由商用。源数据的选择、校正和任务细节见主工程`docs/experiments/`。
