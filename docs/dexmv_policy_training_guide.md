# DexMV Pure TRPO Policy Training Guide

本文说明当前 `DEXMV-PPO` 工程里，纯 DexMV 策略是怎样生成的、训练时使用了哪些数据、强化学习过程是什么，以及后续应该从哪些方向优化。

## 当前训练结论

当前长训练运行的是 **DexMV 自带 MuJoCo 环境 + mjrl TRPO**，不是 Aero-Hand PPO，也没有使用 DexMV 示教数据。

本轮命令：

```bash
cd /home/ll/SRTP/DEXMV-PPO/DexMV/examples
env -u PYTHONPATH \
  PATH=/home/ll/miniconda3/envs/dexmv/bin:$PATH \
  LD_LIBRARY_PATH=/home/ll/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia \
  MUJOCO_PY_MUJOCO_PATH=/home/ll/.mujoco/mujoco210 \
  /home/ll/miniconda3/envs/dexmv/bin/python train.py \
    --cfg configs/dapg-mug-example.yaml \
    USE_DAPG False \
    BC_INIT False \
    TRPO.ENABLED True \
    NUM_ITER 200 \
    NUM_CPU 4 \
    NUM_TRAJ 20 \
    EVAL_RS 10 \
    SAVE_FREQ 25 \
    JOB_NAME pure_trpo_long_cpu_i200_t20 \
    JOB_DIR ./training_log
```

训练输出目录：

```text
DexMV/examples/training_log/rl_relocate-mug-0.8_pure_trpo_long_cpu_i200_t20_seed200/
```

本轮实际结果：

```text
总轮数：200
训练采样最高点：iter 170, sampling_return = -0.75
训练时 evaluation 最高点：iter 59, eval_score = 107.71
最后一轮：iter 199, sampling_return = -16.68, eval_score = -9.70
前 10 轮 eval 平均：29.70
最后 20 轮 eval 平均：-10.16
```

独立 deterministic evaluation，20 个 episode，固定同一批 seed：

```text
policy_25.pickle     mean -5.650   max -3.995
policy_50.pickle     mean -2.375   max 23.635
policy_75.pickle     mean -6.456   max -4.386
policy_100.pickle    mean -13.114  max -10.599
policy_125.pickle    mean -11.481  max -9.125
policy_150.pickle    mean -11.958  max -10.266
policy_175.pickle    mean -11.550  max -9.227
best_policy.pickle   mean -11.399  max -8.743
```

所以本轮保存点里最值得看的不是 `best_policy.pickle`，而是：

```text
DexMV/examples/training_log/rl_relocate-mug-0.8_pure_trpo_long_cpu_i200_t20_seed200/iterations/policy_50.pickle
```

原因是 DexMV 原始 `train_agent.py` 里的 `best_policy` 按训练采样回报选，而不是按 evaluation 回报选；纯 TRPO 的训练采样带噪声，可能把偶发高回报轨迹误判成更好的 deterministic 策略。

可视化脚本：

```bash
./scripts/view_dexmv_policy.sh \
  DexMV/examples/training_log/rl_relocate-mug-0.8_pure_trpo_long_cpu_i200_t20_seed200/iterations/policy_50.pickle \
  500
```

已生成 GIF：

```text
visualizations/dexmv_long_trpo_policy50_relocate_mug.gif
```

## 策略具体是怎么生成的

入口脚本是：

```text
DexMV/examples/train.py
```

核心流程如下：

1. 读取配置文件 `DexMV/examples/configs/dapg-mug-example.yaml`。
2. 命令行覆盖配置，把 `USE_DAPG=False`、`BC_INIT=False`，所以关闭示教预训练和 DAPG。
3. 创建任务环境 `YCBRelocate(has_renderer=False, object_name="mug", object_scale=0.8)`。
4. 创建策略网络 `mjrl.policies.gaussian_mlp.MLP`。
5. 创建 value baseline `mjrl.baselines.mlp_baseline.MLPBaseline`。
6. 创建 TRPO agent。
7. 每一轮从当前策略采样若干条轨迹。
8. 根据环境 reward 计算 return 和 advantage。
9. TRPO 用 KL 约束更新策略网络。
10. 定期保存 `policy_*.pickle`、`baseline_*.pickle` 和最终 `best_policy.pickle`。

当前策略网络结构：

```text
输入：39 维 observation
隐藏层：32, 32
输出：30 维 action mean
动作分布：Gaussian(mean, learned log_std)
```

其中 evaluation 时使用的是策略均值：

```python
action = policy.get_action(obs)[1]["evaluation"]
```

训练采样时使用的是带噪声动作：

```python
action = mean + exp(log_std) * normal_noise
```

## 当前使用了哪些数据集

严格说，本轮纯 TRPO **没有使用离线数据集**。

它使用的是在线强化学习数据：

```text
当前策略 -> MuJoCo 环境 rollout -> observations/actions/rewards -> TRPO 更新
```

每一轮的数据由仿真即时生成。本轮配置：

```text
NUM_TRAJ = 20
EVAL_RS = 10
NUM_ITER = 200
```

也就是说：

```text
每轮训练采样 20 条轨迹
每轮评估采样 10 条轨迹
总共 200 轮
```

这些在线轨迹不会作为固定 dataset 预先存盘再训练，而是每轮采样、每轮更新。

配置文件里原本有：

```yaml
DEMO_FILE: ../demonstrations/relocate-mug.pkl
USE_DAPG: true
```

但当前训练命令显式覆盖为：

```text
USE_DAPG False
BC_INIT False
```

所以 `DEMO_FILE` 没有被读取。

## 环境和 reward 是什么

当前任务类：

```text
DexMV/hand_imitation/env/environments/ycb_relocate_env.py
```

任务目标是让 Adroit hand 抓起 YCB mug，并移动到目标位置。

observation 由四部分组成：

```text
hand_qpos[0:30]
palm_pos - obj_pos
palm_pos - target_pos
obj_pos - target_pos
```

总维度：

```text
30 + 3 + 3 + 3 = 39
```

action 是 30 维，范围被归一化到 `[-1, 1]`，环境内部再映射到 MuJoCo actuator control range。

reward 结构：

```text
-0.1 * palm_to_object_distance
+0.1 contact_bonus
+50 * lift_height
+2.0 lifted_bonus
-0.5 * palm_to_target_distance
-1.5 * object_to_target_distance
+1 / (object_to_target_distance + 0.01) if close enough
```

这个 reward 是 dense reward，适合纯 RL 起步，但也容易出现策略钻 reward 空子，例如短暂抬起但不稳定、评估波动大、动作不自然等。

## TRPO 是怎么强化学习的

每一轮训练中：

1. 用当前策略采样 `NUM_TRAJ` 条轨迹。
2. 对每条轨迹计算折扣回报：

```text
return_t = reward_t + gamma * reward_{t+1} + ...
gamma = 0.995
```

3. 用 baseline 预测 value，计算 advantage。
4. 对 advantage 做标准化。
5. 计算策略梯度。
6. 用共轭梯度近似自然梯度方向。
7. 用 KL 约束限制每次策略更新幅度，避免策略突然崩掉。
8. 如果 surrogate improvement 不好，会 backtracking 缩小步长。

当前 TRPO 的优势是训练相对稳定，适合这种老版 mjrl/DexMV 栈。缺点是采样效率不高，且最终实用策略质量高度依赖 reward、随机化范围和示教/先验。

本轮长训的实际现象是：纯 TRPO 有学习信号，但不稳定。早期 evaluation 出现高峰，后期退化到长期负值。这说明单纯加长训练不能保证策略变好，后续更应该引入 demonstration、curriculum 或 residual prior。

## 训练产物怎么看

主要文件：

```text
results.txt
logs/log.csv
logs/log.pickle
logs/stoc_pol_mean.png
iterations/policy_25.pickle
iterations/policy_50.pickle
...
iterations/best_policy.pickle
```

`results.txt` 每行格式：

```text
iter sampling_return evaluation_return best_sampled_return
```

重要指标：

```text
stoc_pol_mean: 训练采样平均回报
eval_score: evaluation 策略平均回报
kl_dist: 本轮策略更新前后的 KL 距离
VF_error_before / after: baseline 拟合误差
```

当前最应该关注 `eval_score` 和可视化行为，而不是只看 `stoc_pol_mean`。因为训练采样带探索噪声，evaluation 才接近最终部署时的 deterministic 策略。

## 应该如何优化

建议优化顺序：

1. 先把纯 TRPO 作为 baseline。
2. 再开启 DexMV demonstration，跑 BC 或 DAPG。
3. 再做和 Aero-Hand PPO 的桥接或 residual PPO。

具体方向：

### 1. 增加训练长度和采样量

当前长训是：

```text
200 iter x 20 trajectories
```

如果评估曲线还在上升，可以扩到：

```text
500 iter x 50 trajectories
```

但纯 TRPO 很吃样本，长训不一定比引入示教更划算。

### 2. 使用 DexMV demonstration

如果你下载或生成了：

```text
DexMV/demonstrations/relocate-mug.pkl
```

可以开启：

```text
BC_INIT True
USE_DAPG True
```

这会让策略先模仿示教，再用 RL fine-tune。对于灵巧手抓取，通常比纯 RL 更实用。

### 3. 改 reward

当前 reward 已经包含接触、抬升、靠近目标等项。后续可以优化：

```text
提高稳定抓握奖励
惩罚过大动作
惩罚手指抖动
加入 object velocity 稳定项
加入 task success gate
```

这对最终迁移到 Aero-Hand 或硬件非常重要。

### 4. 做 curriculum

先训练简单任务：

```text
固定 mug 初始位置
较近 target
低随机化
```

再逐步增加：

```text
随机初始位置
随机目标位置
不同物体尺度
摩擦和质量扰动
```

这比一开始直接高随机化更稳。

### 5. 引入 residual PPO

最终强融合建议不是“DexMV 训完一个策略，然后 Aero PPO 从头学”，而是：

```text
action = dexmv_prior(obs) + residual_policy(obs)
```

DexMV policy 提供抓取先验，Aero PPO 学残差修正。这样最终决策主体不是单纯 PPO，也不是单纯 DexMV，而是“先验策略 + 任务自适应残差”。

## 数据集可以更改吗

可以。

DexMV 的示教数据由配置项控制：

```text
DEMO_FILE
DEMO_RATIO
BC_INIT
USE_DAPG
```

可以换成：

```text
不同物体的 demonstration
不同质量的 demonstration
人工 retarget 后的 demonstration
从真实手/视频重定向来的 demonstration
你自己用脚本生成的 demo_dataset
```

但必须保证数据字段和 mjrl 期望一致，通常每条 trajectory 至少要包含：

```text
observations
actions
rewards 或 rewards 可重算
env_infos / agent_infos 可选
```

如果要给 Aero-Hand PPO 用，建议不要直接复用 DexMV `.pkl`，而是桥接成统一格式：

```text
demo_dataset.npz
obs
actions
qpos
object_pose
target_pose
contact 或 success 标签
```

## 手的模型可以更改吗

可以，但不是简单替换 mesh 文件就完事。

当前 DexMV 的手模型是：

```text
DexMV/hand_imitation/env/models/assets/adroit/adroit_relocate.xml
```

如果更换手模型，需要同时处理：

```text
actuator 数量
joint 数量
qpos/qvel 维度
action 映射
observation 拼接
site 名称，例如 S_grasp
geom 名称和 contact group
reward 里的 palm/object/target 引用
policy 输入输出维度
已有 demonstration 是否还能对齐
```

如果换成 Aero-Hand，建议不要直接把 Aero-Hand XML 硬塞进 DexMV 环境，而是做中间接口：

```text
DexMV prior pose/action
        ↓
retarget / adapter
        ↓
Aero-Hand action space
        ↓
residual PPO correction
```

这样能保留 DexMV 的手势先验，同时让最终控制适配 Aero-Hand 的真实 actuator 和动力学。

## 当前最推荐的下一步

1. 等本轮纯 TRPO 长训完成，确认 baseline 上限。
2. 下载或生成 `relocate-mug.pkl` demonstration。
3. 跑 DAPG/BC+DAPG，和纯 TRPO 对比。
4. 梳理 DexMV observation/action 到 Aero-Hand observation/action 的映射。
5. 做 residual PPO，而不是只做初始化。

判断融合是否足够强的标准：

```text
最终策略输入里能看到 Aero-Hand 当前状态
最终动作由 DexMV prior 和 residual policy 共同决定
DexMV prior 可以在训练中继续被评估、蒸馏或约束
PPO 不是简单从 DexMV 初始姿态继续训练，而是在每个 step 都利用 DexMV 输出
```

## 最终目标框架：视频先验 + PPO 姿态修正

你的目标不是“DexMV 训练一个完整策略，然后 PPO 接着训练”，而是：

```text
每一步最终姿态 = 视频/DexMV 先验姿态 + PPO 修正量
```

更具体地说：

```text
video
  ↓
DexMV hand pose / retargeting
  ↓
prior_qpos[t]              Aero-Hand 当前观测 obs[t]
  ↓                         ↓
时间对齐 / action adapter     PPO policy
  ↓                         ↓
prior_target_pose[t]       delta_pose[t]
  \                         /
   \                       /
    final_target_pose[t] = prior_target_pose[t] + delta_pose[t]
                    ↓
              Aero-Hand 环境执行
                    ↓
     reward: 抓取成功 + 姿态合理 + 修正平滑 + 稳定接触
                    ↓
              更新 PPO residual policy
```

这个框架里：

```text
DexMV 负责：从视频/手部姿态得到先验姿态轨迹
PPO 负责：在 Aero-Hand 环境中学习每一步该修多少
```

最终执行时，DexMV 先验不是只在训练开始时用一次，而是每个 step 都参与：

```python
prior = prior_dataset["qpos"][t]
delta = ppo_policy(obs)
target = prior + delta_scale * delta
```

第一版建议先做 position residual：

```text
final_qpos_target = prior_qpos[t] + delta_qpos[t]
```

后面再升级为：

```text
final_action = prior_action[t] + residual_action[t]
```

如果 Aero-Hand 控制接口是 actuator command，而不是 qpos target，则需要一个 adapter：

```text
prior_qpos -> PD target / actuator target / normalized action
```

## DAPG 和这个框架的区别

DAPG 是：

```text
视频/示教数据
  ↓
BC 模仿
  ↓
TRPO 强化学习
  ↓
得到一个完整策略 policy(obs)
```

执行时：

```text
action = policy(obs)
```

而你的 Residual PPO 是：

```text
action_or_pose = video_prior[t] + ppo_residual(obs)
```

区别在于：

```text
DAPG：视频示教被“揉进”一个策略里，最后只有一个策略决策。
Residual PPO：视频先验每一步都显式存在，PPO 每一步只学修正量。
```

所以 DAPG 可以作为后续增强，但不应该替代你的主框架。

推荐优先级：

```text
第一阶段：DexMV retarget -> prior_qpos trajectory
第二阶段：prior_qpos -> demo_dataset.npz
第三阶段：Aero-Hand PPO 输出 delta_qpos
第四阶段：final_qpos = prior_qpos + delta_qpos
第五阶段：加入 pose loss / smooth loss / contact reward / task reward
第六阶段：如果 prior 太差，再考虑 BC/DAPG 训练一个 prior policy
```

## 本次已完成的 DexMV 重定向

本次使用的是 DexMV 自带的 processed 示例，不是你自己的原始视频：

```text
DexMV/examples/retargeting_source/relocate_mustard_example_seq/
```

输入包含两类数据：

```text
hand_pose/results_global_*.npy
hand_pose/joints_*.npy
```

其中：

```text
results_global_*.npy：每帧人手全局位姿
joints_*.npy：每帧人手关键点 / 关节位置
```

运行命令：

```bash
cd /home/ll/SRTP/DEXMV-PPO/DexMV/examples
env -u PYTHONPATH \
  PATH=/home/ll/miniconda3/envs/dexmv/bin:$PATH \
  LD_LIBRARY_PATH=/home/ll/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia \
  MUJOCO_PY_MUJOCO_PATH=/home/ll/.mujoco/mujoco210 \
  /home/ll/miniconda3/envs/dexmv/bin/python retarget_human_hand.py \
    --hand_dir=./retargeting_source/relocate_mustard_example_seq/hand_pose \
    --output_file=/home/ll/SRTP/DEXMV-PPO/artifacts/dexmv_retarget/example_retargeting.pkl
```

为了跑通 retargeting，额外补齐了依赖：

```text
nlopt
absl-py
lxml
```

输出文件：

```text
artifacts/dexmv_retarget/example_retargeting.pkl
artifacts/dexmv_retarget/example_retargeting.npy
artifacts/dexmv_retarget/example_retargeting_summary.npz
```

输出内容：

```text
shape = (122, 30)
含义 = 122 帧，每帧 30 维 Adroit hand robot qpos
```

这一步得到的是：

```text
human hand pose sequence -> Adroit hand joint position sequence
```

它还不是最终 Aero-Hand PPO 数据集。下一步需要桥接：

```text
Adroit qpos(30)
  ↓
retarget / adapter
  ↓
Aero-Hand qpos/action
  ↓
demo_dataset.npz
```

## 还需要哪些数据

如果只是验证 DexMV retargeting，本地自带示例数据已经够了。

如果要做你的最终系统，还需要至少一套你自己的任务数据：

```text
1. 你的视频，最好是目标物体清晰、手和物体同时可见
2. 视频对应的手部姿态估计结果
3. 物体姿态估计结果
4. 相机外参 / 坐标系标定
5. 任务目标定义，例如最终物体位置、抓取成功判定
```

注意：当前 `dexmv-sim` 仓库不负责从原始视频直接估计手姿态。它的 retargeting 示例假设你已经有：

```text
hand_pose/*.npy
object_pose/*.npy
```

也就是说，完整视频链路应该是：

```text
原始视频
  ↓
手部/物体姿态估计算法
  ↓
hand_pose/*.npy + object_pose/*.npy
  ↓
DexMV retargeting
  ↓
robot qpos trajectory
  ↓
DEXMV-PPO bridge
  ↓
demo_dataset.npz
  ↓
Aero-Hand Residual PPO
```

如果你后续想用 BC/DAPG，还需要 DexMV demonstration 格式的数据：

```text
DexMV/demonstrations/*.pkl
```

但对于第一版 Residual PPO，不一定需要 DAPG demonstration。第一版最小数据可以是：

```text
prior_qpos[t]
object_pose[t]
target_pose
time_index
```

再由 Aero-Hand 环境在线产生：

```text
obs[t]
reward[t]
success/contact 信息
```

## 当前最小可执行下一步

下一步不是继续 TRPO，而是写桥接脚本：

```text
artifacts/dexmv_retarget/example_retargeting.npy
  ↓
DexMV/Adroit qpos -> Aero-Hand qpos/action adapter
  ↓
demo_dataset.npz
```

然后在 Aero-Hand PPO 里把 action head 改成 residual：

```text
delta = policy(obs)
target = prior[t] + delta_scale * delta
```

并加入奖励：

```text
task_reward              抓住 / 抬起 / 稳定
pose_prior_reward        不要无意义偏离视频先验
residual_l2_penalty      修正量小
smoothness_penalty       修正别抖
contact_reward           合理接触
```
