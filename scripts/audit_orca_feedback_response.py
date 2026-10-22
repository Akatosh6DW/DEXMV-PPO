"""Exploratory response audit on the four saved B0/B5 cases per policy.

This is not an extra significance test or a replacement for an open-loop replay
ablation. It checks whether changed physical state changes the issued targets.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(nominal, pushed):
    rows = []
    for i in range(len(nominal["qpos"])):
        prefix_q = float(np.max(np.abs(nominal["qpos"][i, :200] - pushed["qpos"][i, :200])))
        prefix_a = float(np.max(np.abs(nominal["policy_action"][i, :201] - pushed["policy_action"][i, :201])))
        pre_valid = bool(nominal["alive"][i, 199] and pushed["alive"][i, 199]
                         and nominal["hold"][i, 199] > 0 and pushed["hold"][i, 199] > 0)
        active = nominal["active"][i, 201:224] & pushed["active"][i, 201:224]
        target_delta = pushed["desired_ctrl"][i, 201:224] - nominal["desired_ctrl"][i, 201:224]
        residual_delta = pushed["requested_residual"][i, 201:224] - nominal["requested_residual"][i, 201:224]
        rms = float(np.sqrt(np.mean(target_delta[active]**2))) if active.any() else None
        rrms = float(np.sqrt(np.mean(residual_delta[active]**2))) if active.any() else None
        matched = prefix_q <= 1e-6 and prefix_a <= 1e-6
        rows.append(dict(pre_push_valid=pre_valid, matched_prefix=matched,
                         prefix_qpos_max_difference=prefix_q, prefix_action_max_difference=prefix_a,
                         paired_active_steps=int(active.sum()), target_response_rms_rad=rms,
                         residual_response_rms_rad=rrms,
                         responding=bool(pre_valid and matched and rms is not None and rms > 1e-6)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    rows, missing = [], []
    for record in protocol["jobs"]:
        if not record["name"].startswith("eval/B5_push/"):
            continue
        pushed = root / record["name"]
        nominal = root / "eval/B0_nominal" / pushed.name
        if not all((d / "summary.json").exists() for d in (nominal, pushed)):
            missing.append(record["name"])
            continue
        configs = [json.loads((d / "run_config.json").read_text()) for d in (nominal, pushed)]
        for key in ("method", "seed", "arm", "variant", "checkpoint_sha256", "reference_sha256"):
            assert configs[0].get(key) == configs[1].get(key)
        traces = []
        for directory in (nominal, pushed):
            with np.load(directory / "step_000000000/preview_trajectories.npz") as z:
                traces.append({key: z[key] for key in z.files})
        assert np.max(np.abs(traces[0]["push"])) == 0
        magnitudes = np.linalg.norm(traces[1]["push"], axis=-1)
        expected = np.zeros_like(magnitudes)
        expected[:, 200:204] = .15
        np.testing.assert_allclose(magnitudes, expected, atol=1e-6)
        method = pushed.name.rsplit("_s", 1)[0] if "_s" in pushed.name else pushed.name
        for seed, result in zip(protocol["test_seeds"][:4], compare(*traces)):
            if method.startswith("zero_"):
                assert not result["responding"]
                assert result["target_response_rms_rad"] in (None, 0.)
            rows.append(dict(method=method, train_seed=configs[0]["seed"], test_seed=seed, **result))
    output = root / "report/feedback_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output / "cases.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = []
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        valid = [r for r in selected if r["pre_push_valid"] and r["matched_prefix"] and r["paired_active_steps"]]
        groups.append(dict(method=method, saved_pairs=len(selected), interpretable_pairs=len(valid),
                           responding_pairs=sum(r["responding"] for r in valid),
                           mean_target_response_rms_rad=float(np.mean([r["target_response_rms_rad"] for r in valid])) if valid else None,
                           mean_residual_response_rms_rad=float(np.mean([r["residual_response_rms_rad"] for r in valid])) if valid else None))
    result = dict(complete=not missing, missing=missing, groups=groups, paired_cases=len(rows),
                  exploratory=True, additional_hypothesis_tests=0,
                  threshold_rad=1e-6, comparison_window_action_indices=[201, 224],
                  note="First four saved test cases only; matched before push. Demonstrates state-dependent output, not the causal performance benefit of feedback over a learned open-loop offset.",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# 外力扰动后的动作响应诊断", "", f"状态：{'完成' if not missing else '未完成'}；固定保存案例对数：{len(rows)}。",
             "这是探索性机制检查，不加入主假设检验，不代替完整200初态的性能统计。",
             "比较同一策略、同一初态B0与B5的共同存活时段，先检查施力前轨迹和动作是否匹配。",
             "响应窗口为10.05..11.15秒发出的动作；要求施力前双方有效抓握。", "",
             "| 方法 | 保存案例对 | 可解释案例对 | 有目标角响应的案例对 | 目标角响应RMS(rad) | 修正响应RMS(rad) |",
             "|---|---|---|---|---|---|"]
    for row in groups:
        target = row["mean_target_response_rms_rad"]
        residual = row["mean_residual_response_rms_rad"]
        lines.append(f"| {row['method']} | {row['saved_pairs']} | {row['interpretable_pairs']} | "
                     f"{row['responding_pairs']} | {target} | {residual} |")
    lines += ["", "有响应说明控制输出不是完全固定的时间表；不代表响应方向正确或提高了成功率。",
              "进一步隔离闭环反馈收益，需要将学到的修正轨迹固定回放，和闭环策略做预先设计的对照。",
              "纯PPO/BC的修正字段恒零，因为它们输出全目标；应看目标角响应，不能说它们没有反馈。"]
    (output / "results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(complete=not missing, paired_cases=len(rows), missing=len(missing))))
    if missing and not args.allow_partial:
        raise RuntimeError("Feedback diagnostic pairs incomplete")


if __name__ == "__main__":
    main()
