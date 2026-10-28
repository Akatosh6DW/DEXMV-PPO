# 双人协作约定

仓库所有者：`Akatosh6DW`；共同开发者：`jb6396892-byte`。主仓库、Aero-Hand、dexmv-sim 已分别发送 write 权限邀请，对方须在 GitHub 接受。无需共享密码、Token 或 SSH 私钥。

## 工作步骤

1. 从最新 main 建自己的分支，例如 `akatosh/s13-reward`、`jb/s13-demonstrations`。
2. 一人一次实验使用独立目录，例如 `outputs/akatosh/S13_001/`，不共用 checkpoint/日志目录。
3. 修改前记录任务、成功判据、输入 SHA、奖励版本、预算、训练/开发/测试种子及预期机制。正式测试集不参与调参。
4. 先跑接口测试、短训练及独立评估，再开长训练。报告失败和异常，不只挑成功视频。
5. 提交代码、配置、结果表和分析；大型轨迹走版本化 Release，权重和样本按 SHA 去重。不要 `git add -f artifacts` 或提交整个环境。
6. 提交 Pull Request，由另一人审查。先合并子仓库 PR，再在主仓库更新子模块指针，保证所引用提交已推送。

```bash
git switch main
git pull --ff-only
git submodule update --init --recursive
git switch -c jb/s13-demonstrations
# 修改、测试，再精确添加自己改动的文件
git push -u origin jb/s13-demonstrations
gh pr create
```

子模块通常处于 detached HEAD，修改前在子仓库内 `git switch -c jb/任务名`。Aero-Hand 与 DexMV 的推送目标应在 Akatosh6DW 名下；DexMV-Learn 目前只读跟踪原作者提交，不向原作者仓库推送。确需修改它时先创建自有 fork，再审查子模块 URL 改动。

## 审查要求

- 不更改既有协议来提高已公布成绩；新奖励、新参考、新样本必须成为新实验或明确消融。
- BC 后 PPO 与执行时残差是不同机制，报告中写动作公式及参考是否参与每一步。
- 比较条件、交互预算与测试初态一致，另外列 BC/校正数据成本；记录逐种子结果和失败分母。
- 软体网格失效、穿透、支撑未撤、非目标部位支撑必须单列；不要只用 return 宣称成功。
- 权重热启动与完整训练状态恢复分开说明。
- PKL 只加载可信来源；新二进制须附 SHA 和来源。源视频、许可证不清晰的数据、凭据不得随 PR 上传。

CI 检查发布完整性和 CPU 接口，不替代 GPU 长训练或实机验证。CODEOWNERS 指向两位开发者，主分支采用 PR 审查流程；管理员紧急修复仍应留记录。
