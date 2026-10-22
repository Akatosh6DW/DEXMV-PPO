"""Audit recorded numerical failures and report conservative success bounds."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/orca_cylinder_s02'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    invalid_by_job, events_out, missing = {}, [], []
    for path in sorted((OUT/'numerical_recovery_plans').glob('*.json')):
        plan = json.loads(path.read_text())
        for name, expected in plan['sources'].items():
            assert digest(ROOT/name) == expected, name
        assert plan['v2_sha256'] == digest(OUT/'terminal_tail_v2.json')
        diagnostic = ROOT/plan['diagnostic']
        assert digest(diagnostic/'diagnosis.json') == plan['diagnosis_sha256']
        assert digest(diagnostic/'prefix.npz') == plan['prefix_sha256']
        assert digest(OUT/'diagnostics/numerical_validation/validation.json') == plan['validation_sha256']
        run = OUT/plan['job']
        if not (run/'summary.json').exists():
            missing.append(plan['job'])
            continue
        config = json.loads((run/'run_config.json').read_text())
        assert config['numerical_recovery_plan_sha256'] == digest(path)
        assert config['job_sha256'] == plan['job_sha256']
        for name, expected in plan['sources'].items():
            if name.endswith('audit_cylinder_s02_numerical.py'):
                continue
            assert config['source_sha256'][Path(name).name] == expected
            assert digest(run/'sources'/Path(name).name) == expected
        evidence = json.loads((run/'numerical_failures.json').read_text())
        assert evidence['plan_sha256'] == digest(path)
        episodes = json.loads((run/'step_000000000/episodes.json').read_text())
        bad = set(evidence['invalid_seeds'])
        assert bad and bad == {e['seed'] for e in evidence['events']}
        for event in evidence['events']:
            assert digest(run/'physics_failure_states'/event['state_file']) == event['state_sha256']
            index = event['batch_index'][0]
            assert episodes[index]['seed'] == event['seed']
            assert not episodes[index]['success30'] and not episodes[index]['clean_success30']
            events_out.append(dict(job=plan['job'], seed=event['seed'],
                                   failure_time=event['incoming_time'], step_index=event['step_index'],
                                   cause=event['error'], metrics_censored=True))
        with np.load(run/'numerical_prefix.npz') as actual, np.load(diagnostic/'prefix.npz') as old:
            np.testing.assert_array_equal(actual['alive'], old['alive'])
            active = np.concatenate([np.ones((200, 1), bool), old['alive'][:, :-1]], axis=1)
            for name in ('qpos', 'qvel'):
                np.testing.assert_array_equal(actual[name][active], old[name][active])
        invalid_by_job[plan['job']] = bad
    protocol = json.loads((OUT/'protocol.json').read_text())
    grouped, completed = {}, 0
    for record in protocol['jobs']:
        if not record['name'].startswith('eval/'):
            continue
        run = OUT/record['name']
        if not (run/'summary.json').exists():
            continue
        completed += 1
        label = record['name'].split('/')[-1]
        method = label.rsplit('_s', 1)[0] if '_s' in label else label
        benchmark = record['name'].split('/')[1]
        bucket = grouped.setdefault((benchmark, method), dict(episodes=0, successes=0, invalid=0))
        episodes = json.loads((run/'step_000000000/episodes.json').read_text())
        bucket['episodes'] += len(episodes)
        bucket['successes'] += sum(e['clean_success30'] for e in episodes)
        bucket['invalid'] += len(invalid_by_job.get(record['name'], set()))
    bounds = [dict(benchmark=b, method=m, **v,
                   lower_success=v['successes']/v['episodes'],
                   upper_success=(v['successes']+v['invalid'])/v['episodes'])
              for (b, m), v in sorted(grouped.items())]
    complete = completed == 388 and not missing
    result = dict(complete=complete, completed_evaluations=completed, missing=missing,
                  recovered_jobs=len(invalid_by_job), invalid_episodes=len(events_out), bounds=bounds)
    (OUT/'numerical_recovery_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    if not complete and not args.allow_partial:
        raise RuntimeError('Numerical-failure audit requires the complete evaluation matrix')
    if complete:
        report = OUT/'report'
        report.mkdir(exist_ok=True)
        for name, rows in [('numerical_failures.csv', events_out), ('numerical_failure_bounds.csv', bounds)]:
            with (report/name).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        lines = ['## 有效回合内的数值异常', '',
                 f'共 {len(events_out)} 个正式回合发生记录在案的数值异常；全部保留在原测试分母中。',
                 '这不同于掉落后尾段发散。异常发生前照常仿真，异常步起停止该回合并保存警告及输入状态。',
                 '主表将它们保守计为未成功，不更换物理参数，不删除初态，也不声称这证明了现实中的抓取失败。',
                 '下面同时给出把所有异常都当作成功时的上界；这是数值异常敏感性边界，不是统计置信区间。', '',
                 '| 条件/方法 | 异常/总回合 | 保守成功率 | 全部异常算成功的上界 |',
                 '|---|---:|---:|---:|']
        for row in bounds:
            if row['invalid']:
                lines.append(f"| {row['benchmark']}/{row['method']} | {row['invalid']}/{row['episodes']} | "
                             f"{row['lower_success']*100:.2f}% | {row['upper_success']*100:.2f}% |")
        lines += ['', '异常回合的保持时间、接触力等只描述出错前的有效前缀，属于截断数据，不能当完整轨迹比较。',
                  '清单见 `report/numerical_failures.csv`，边界见 `report/numerical_failure_bounds.csv`。',
                  '其余有效轨迹不改变；独立正常批次 200 回合统计及预览轨迹完全一致，异常批次前缀另行逐项核对。',
                  '这项异常处理是执行期间补充的规则；涉及异常条件的统计结论需结合上述边界解释。']
        (OUT/'numerical_recovery.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'bounds'}), flush=True)


if __name__ == '__main__':
    main()
