"""Reuse frozen S02 statistics and explain the cylinder experiment separately."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import report_orca_suite

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/orca_cylinder_s02'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT)
    args = parser.parse_args()
    if args.root.resolve() != OUT.resolve():
        parser.error('This explanation is bound to the frozen cylinder S02 study')
    sys.argv[1:] = ['--root', str(OUT)]
    # The original CLI performs all coverage checks and exact paired statistics.
    report_orca_suite.main()
    stats = json.loads((OUT/'report/statistics.json').read_text())
    if not stats['complete']:
        raise ValueError('The final cylinder explanation requires the complete matrix')
    aggregate, contrasts = stats['aggregate'], stats['contrasts']
    names = report_orca_suite.NAMES
    nominal = {r['method']: r for r in aggregate if r['benchmark'] == 'B0_nominal'}
    lines = ['# 圆柱抓持完整实验', '',
             '本实验把原 S02 方块换成同质量的圆柱，保留 PPO、BC 后 PPO、参考加残差 PPO 和原训练配方。',
             '50 次训练和 388 批评估均已完成，正式测试共 77600 回合。以下不是只挑选成功视频的演示。', '',
             '## 任务到底是什么', '',
             'Orca 手的基座固定，没有可移动机械臂；1 个腕关节和 16 个手指关节仍可控。',
             '直径和高度均 53 mm、质量 35 g 的圆柱先由托板支撑；',
             '手指合拢后托板撤走，策略要维持物体不掉落。每回合 40 秒、每秒控制 20 次。',
             '这不是机械臂从桌面任意位置寻找物体，也不是自由手腕的通用抓取。',
             '主要成功标准为连续有效抓持至少 30 秒且结束时仍有效，还要通过原生物理子步接触审计。',
             '手掌等禁止部位不能偷托住物体，撤托后穿透不得超过 3 mm。', '',
             '## 用什么学习', '',
             '使用 DexMV 官方已处理的番茄罐示范序列 96，处理后帧 182 到 271。',
             '不是重新运行原始 RGB 感知，也没有复用作者的策略权重。原始 MP4 与该序列的对应未独立确认。',
             '源物体约 70 mm 宽、104 mm 高，分别缩放径向和高度，映射到 53 mm 圆柱。',
             '把指尖放到物体坐标系中，匹配机器人实际指腹碰撞体，通过逆运动学求关节参考。',
             '逆运动学就是：已知指腹要去哪里，求哪些关节角度可以让它尽量到那里。',
             '映射并不完美：末帧两指几何误差约 14.13 和 6.69 mm，不能称为完全复刻人手。',
             '参考在 64 个采集初态中产生 12 条完整合格回放；只用这些成功回放做 BC，失败片段不混入。',
             '9 条回放训练、3 条验证。它们来自同一参考，不是 12 段独立人类视频。', '',
             '## 三种主方法', '',
             '- 纯 PPO：通过仿真探索和奖励学习，每步直接决定绝对控制目标。',
             '- BC 后 PPO：先用行为克隆模仿成功示范，再由 PPO 更新整个策略；执行时不叠加视频动作。',
             '- 参考加残差 PPO：每步都有参考动作，PPO 根据当前状态输出小修正，两者相加后执行。',
             '  修正请求限幅为每关节 0.08 rad，约 4.58 度；另有执行器限位和控制变化限制。',
             'BC 是监督学习，不靠自己试错得分；PPO 是强化学习，靠交互奖励改策略；',
             '残差描述的是执行结构，不是另一种强化学习算法。本实验的残差仍由 PPO 学习。', '',
             '## 正常条件的结果', '',
             '| 方法 | 清洁成功率 | 95%区间 | 原始成功率 | 最长保持均值 |',
             '|---|---:|---:|---:|---:|']
    for method in ('ppo', 'bc', 'guided', 'zero_video', 'residual', 'linear', 'unrefined',
                   'zero_linear', 'zero_unrefined', 'half_residual'):
        r = nominal[method]
        lines.append(f"| {names[method]} | {r['clean_success30']*100:.2f}% | "
                     f"[{r['ci_low']*100:.2f}%, {r['ci_high']*100:.2f}%] | "
                     f"{r['raw_success30']*100:.2f}% | {r['mean_hold']:.2f} 秒 |")
    lines += ['', '每个学习方法有 10 个独立训练种子、每个策略 200 个测试初态。',
              '零修正参考只有一个固定控制器，评估 200 次；不能把它复制成 10 个独立策略来夸大样本量。', '',
              '边界结果需谨慎：全成功或全失败时，经验 bootstrap 区间会退化成 [100%,100%] 或 [0%,0%]。',
              '它只反映这批观测，不能证明未来必然成功或永不成功；2000 个策略/初态组合也不等于 2000 个不同初态。', '',
              '## 扰动后的表现', '', '| 条件 | 纯 PPO | BC 后 PPO | 参考加残差 PPO |', '|---|---:|---:|---:|']
    for benchmark in ('B0_nominal', 'B1_position', 'B2_yaw', 'B3_friction', 'B4_mass', 'B5_push'):
        values = [next(r for r in aggregate if r['benchmark'] == benchmark and r['method'] == m)
                  for m in ('ppo', 'guided', 'residual')]
        lines.append('| '+benchmark+' | '+' | '.join(f"{r['clean_success30']*100:.2f}%" for r in values)+' |')
    lines += ['', 'B1 是位置扰动，B2 是竖直轴旋转，B3 是低摩擦，B4 是双倍质量，B5 是侧向推力。',
              '圆柱绕自身轴旋转不会改变碰撞形状，所以 B2 不能证明复杂姿态泛化。', '',
              '## 可以下什么结论', '']
    for r in contrasts:
        if r['benchmark'] != 'B0_nominal':
            continue
        supported = r['supports_primary_positive_difference']
        lines.append(f"- {r['comparison']}：差值 {r['difference_pp']:+.2f} 个百分点，"
                     f"95%区间 [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]，"
                     f"Holm-5 校正 p={r['primary_holm_5_p']:.4f}。"
                     + ('支持本基准正常条件下的正向差异。' if supported else
                        '不能据此宣称该正向优势已得到严格统计支持；不代表两者相等。'))
    lines += ['', '区间对训练种子和共享初态同时重采样；显著性检验对同种子的配对差值做双侧检验。',
              '训练种子只有 10 个，全部 30 项比较同时校正时最小可达 p 不足以通过阈值，',
              '不能把个别条件的优势写成所有场景普遍显著更强。失败种子和失败视频全部保留。', '',
              '## 过程与限制', '',
              '原 S02 的 49 项冻结源码/输入均未改变。新脚本只是接入圆柱模型和新示范，',
              '每个方法仍使用原配方和最终检查点；没有根据本次正式测试换样本、调奖励或挑权重。',
              '新数据准备尝试了两条源序列，并修复过把定位标记误当碰撞指腹的问题，完整失败记录仍在。',
              '这次校正是接触几何校正，不是原方块参考中的仿真偏置优化；准备方法和总成本不完全等价。',
              '物体状态来自仿真，手指自碰撞仍关闭，没有硬件试验，也没有证明对新视频、新物体泛化。', '',
              '## 文件怎么读', '',
              '- `report/aggregate.csv`：每条件、每方法的汇总及置信区间。',
              '- `report/per_seed.csv`：逐训练种子数据，避免平均数掩盖不稳定性。',
              '- `report/episodes.csv`：77600 个正式回合的成功、保持、穿透和力指标。',
              '- `report/paired.csv`：配对差值、区间和多重比较校正。',
              '- `report/learning_curves.csv`：训练过程开发集数据，不等于正式测试成绩。',
              '- `train/<方法>_s<种子>/params.pkl`：最终权重；guided 目录还有 `bc_params.pkl`。',
              '- `inputs/reference.npz`：每步参考关节目标，不是神经网络权重。',
              '- `inputs/demos/trajectories.npz`：BC 使用的观测、动作和有效步数组。',
              '- `videos/index.md`：固定初态的手心视角视频，不按成功率挑选。',
              '- `audit.json`：哈希、同初态配对、统计及预览轨迹重算审计。', '',
              '详细准备记录见 `docs/experiments/cylinder_s02_execution.md`。']
    (OUT/'report/results.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable, str(ROOT/'scripts/plot_orca_suite.py'), '--root', str(OUT)], check=True)
    print('Cylinder explanation and figures completed:', OUT/'report/results.md')


if __name__ == '__main__':
    main()
