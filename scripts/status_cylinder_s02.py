"""Write live progress from completed jobs, never treating a partial run as done."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/orca_cylinder_s02'


def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    completed, active = [], []
    for item in protocol['jobs']:
        directory = OUT/item['name']
        if not (directory/'summary.json').exists():
            if directory.exists():
                active.append(item['name'])
            continue
        summary = json.loads((directory/'summary.json').read_text())
        with (directory/'metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        values = [float(r['success30']) for r in rows]
        measured = dict(first=values[0], last=values[-1], max=max(values),
                        best_step=int(rows[values.index(max(values))]['step']))
        if any(summary[key] != value for key, value in measured.items()):
            raise ValueError('Summary disagrees with metrics: '+item['name'])
        if summary['status'] != 'completed' or summary['job_sha256'] != item['sha256']:
            raise ValueError('Invalid job identity')
        completed.append(dict(name=item['name'], **summary))
    training = [r for r in completed if r['name'].startswith('train/')]
    evaluation = [r for r in completed if r['name'].startswith('eval/')]
    status = dict(updated_utc=datetime.now(timezone.utc).isoformat(), train_completed=len(training),
                  train_total=50, eval_completed=len(evaluation), eval_total=388,
                  formal_episodes=len(evaluation)*200, active_or_incomplete=active, completed=completed)
    (OUT/'progress.json').write_text(json.dumps(status, indent=2)+'\n')
    lines = ['# 圆柱 S02 实时进度', '', '更新时间 UTC：'+status['updated_utc'], '',
             f"已完成训练 {len(training)}/50，正式评估 {len(evaluation)}/388，测试回合 {len(evaluation)*200}/77600。",
             '以下训练成绩只来自 8 个开发初态，不是正式测试成功率。训练中不根据这些成绩改变本轮配方。', '',
             '正在运行或尚未完整结束：'+', '.join(active), '',
             '| 训练 | 初值 | 最终值 | 最大值 | 最大值对应步数 |', '|---|---:|---:|---:|---:|']
    for row in training:
        lines.append(f"| {row['name']} | {row['first']*100:.1f}% | {row['last']*100:.1f}% | "
                     f"{row['max']*100:.1f}% | {row['best_step']} |")
    lines += ['', '各作业完整日志在 `logs/`，完整冻结协议在 `protocol.json`。',
              '完整实验结论须等待全部正式评价和审计，不能由本进度表代替。']
    (OUT/'progress.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key: value for key, value in status.items() if key != 'completed'}))
    return len(training) == 50 and len(evaluation) == 388


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        complete = main()
        if complete or not args.watch:
            break
        time.sleep(30)
