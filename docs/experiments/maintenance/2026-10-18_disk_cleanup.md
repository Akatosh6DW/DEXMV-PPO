# 磁盘清理记录（第一类：不丢失内容）

日期：2026-10-18。系统盘可用空间从 **1.6 GB 增加到 34 GB**。这次没有删除任何实验结果、权重、参考或报告，也没有修改任何实验文件的内容。

## 1. 做了什么

| # | 操作 | 释放空间 | 依据 |
|---|---|---:|---|
| 1 | 删除 `.git/objects` 中 24 小时之前的 29 个 `tmp_pack_*`、`tmp_obj_*` | 19.29 GB | 这些是中断的 git 写入留下的残留，git 自己标记为 garbage，没有被任何引用使用。删除前后可达对象数都是 34980 |
| 2 | 在 `.git/info/exclude`（只在本机生效）中加入 `/artifacts/`、`/deliverables/` | 防止再次堆积 | 见第 2 节 |
| 3 | 把逐字节相同的数据文件改成硬链接，共 4011 个文件、2035 组 | 12.44 GB | 范围见下 |
| 4 | 删除 VS Code 24 小时之前的扩展安装包缓存（27 个文件）；运行 `conda clean --all` | 1.05 GB + 约 0.8 GB | 都是可以重新下载的缓存 |

第 3 项的范围：
- 只处理 `artifacts/` 和 `deliverables/` 下的 `.npz`、`.mjb`、`.stl`、`.obj` 文件。
- 跳过 `artifacts/orca_benchmark_suite/S09/`（后台训练正在使用）和 24 小时内改动过的文件。
- 每组保留最早的那份（同时间时优先保留 `artifacts/` 下的原件）。替换前逐字节比对，替换用"临时硬链接 + 原子 rename"完成。

## 2. 为什么 `.git` 会有 28 GB

主仓库只有 3 个提交、8 个文件。占用来自 VS Code 中的 Codex 插件：

- **快照**：它每轮对话给整个工作区做一次 git 快照（`refs/codex/turn-diffs/checkpoints/*`），用于撤销。快照遵守 `.gitignore`，但 `.gitignore` 没有排除 zip、tar、pkl，所以 1.7 GB 的资料包 zip、OakInk tar 等都被写了进去。
- **残留**：大于 512 MB 的文件会先写成 `tmp_pack_*`。磁盘满时写入中断，残留文件就留在原地，于是磁盘更满。

加入第 2 项的排除规则后，新快照会跳过这两个目录。以后如果要提交其中某个小文件，用 `git add -f <路径>`。

## 3. 核验

- 4011 个目标文件都存在，并且和各自的保留文件是同一个 inode；没有残留的 `.dedupe_tmp` 文件。
- 2035 份保留文件重新计算 SHA256，全部与清理前的记录一致。资料包里的哈希清单因此依然有效。
- `aero_rl`（JAX 0.9.0.1、MuJoCo 3.5.0、Brax 0.14.1）和 `dexmv`（numpy、torch、nlopt）都能正常导入；后台的 S09 训练进程没有受影响。

## 4. 使用硬链接后要注意

- 同一组硬链接共享同一份数据。如果对其中一个路径**原地改写**（打开后覆盖写入），同组所有路径会一起改变。本项目的输出脚本不会覆盖已有结果，而且这次只处理了上述 4 种二进制数据格式。
- 如果真需要单独修改某个文件，先断开链接：`cp --remove-destination <保留文件> <目标>`，再修改目标。
- 拷到移动硬盘时，用 `rsync -aH` 可以保留硬链接；普通复制会恢复成多份独立文件，内容不受影响。

## 5. 记录文件

都在 `artifacts/maintenance/disk_cleanup_20261018/` 下：

| 文件 | 内容 |
|---|---|
| `git_tmp_deleted.tsv` | 删除的 git 残留文件：时间、大小、路径 |
| `duplicate_groups.json` | 清理前扫描出的全部重复组（含 SHA256） |
| `hardlink_log.jsonl` | 每个被替换文件：保留文件、大小、哈希、原修改时间和权限 |
| `vscode_vsix_deleted.tsv`、`conda_clean.log` | 缓存清理记录 |

执行脚本：[dedupe_artifacts_hardlink.py](../../../scripts/dedupe_artifacts_hardlink.py)。

## 6. 尚未处理（需要另行确认）

- Codex 的 3 个旧快照引用，以及今天 17:21 留下的一个 307 MB 残留：约 9.3 GB。需要等关掉 Codex 后，删除引用再运行 `git gc`。
- 两个重复的资料包 zip（3.2 GB）、OakInk 瓶盖源数据 tar（2.7 GB）、Aero 手阶段剩余的中间检查点（≤ 2.8 GB）。
- 可以移到外部存储：`artifacts/orca_formal/F01_workflow_r2`（21 GB）、`CAP_F01*`（4.1 GB）、`orca_cylinder_raw_video_20261009`（1.3 GB）。
