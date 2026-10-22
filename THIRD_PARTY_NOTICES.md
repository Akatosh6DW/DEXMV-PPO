# 第三方来源与许可

## Orca模型

来源：[orcahand/orcahand_description](https://github.com/orcahand/orcahand_description)。本机来源仓库提交为`9fa90c180c30b593df43c3b0f8499ab5ee866649`；本工程导入目录中的266个STL文件与该来源逐文件SHA-256一致。

许可：MIT，Copyright (c) 2025 ORCA。完整许可保存在`Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/orca_hand/xmls/orcahand_description/LICENSE`。本工程增加了训练任务、简化碰撞体、场景和模型显示逻辑，源模型与修改后的训练表示分别保存。

发布场景中的Orca碰撞代理由该模型几何拟合得到；源CAD和拟合代码保留上述署名。

## 子工程与学习工具

- Aero-Hand：子仓库保留其原有许可、作者信息及第三方mujoco_playground来源。
- DexMV / DexMV-Learn：分别保留子模块中的原许可；使用论文和代码时按其要求引用。
- MuJoCo、JAX、Brax、SciPy等运行依赖沿用各自许可，不由主仓库重新授权。

## 视频及示范数据

本项目使用的HIC、DexMV处理后示范和OakInk来源见各实验协议。上传的参考是经过筛选、机械手重定向或几何校正后的数值序列，原始视频不随本次发布上传。

OakInk相关原始数据与派生标注存在不同许可范围，原始数据CC-BY-SA 4.0、相关派生关节点数据CC-BY-NC 4.0的使用说明保留在对应任务文档。不得因本仓库可访问而将这些数据或其派生内容默认为可商用；后续数据使用应核对具体下载包的原许可和署名要求。

本说明是来源清单，不替代各来源的完整许可文本。
