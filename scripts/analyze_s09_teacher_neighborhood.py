"""Verify teacher perturbation data and distinguish feedback from replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_s09_bc_confirmation import write_csv
from analyze_s09_feedback_perturbation import paired_rows
from probe_s09_feedback_perturbation import ROOT
from retarget_s09_reference import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    read = lambda p: json.loads(p.read_text())
    summary, protocol, rows = [read(root / n) for n in ("summary.json", "protocol.json", "results.json")]
    assert summary["status"] == "completed" and not summary["smoke"]
    assert summary["cases"] == len(rows) == 28
    assert summary["results_sha256"] == digest(root / "results.json")
    assert summary["protocol_sha256"] == digest(root / "protocol.json")
    intervention = protocol.get("intervention_step", 40)
    assert protocol["prefix_steps"] == 40 and protocol["suffix_steps"] == 800 - intervention
    assert protocol["seeds"] == [90531, 90540]
    assert protocol["additional_policy_updates"] == 0 and protocol["inherited_online_steps"] == 1310720
    assert digest(ROOT / protocol["teacher_checkpoint"]) == protocol["teacher_checkpoint_sha256"]
    for name, sha in protocol["execution_sources"].items():
        assert digest(root / "sources" / name) == sha
    expected = {(seed, label, replay) for seed in protocol["seeds"]
                for label, _ in protocol["impulses"] for replay in (False, True)}
    assert {(r["seed"], r["impulse"], r["replay"]) for r in rows} == expected
    prefixes = {}
    for seed in protocol["seeds"]:
        with np.load(root / f"prefix_seed{seed}.npz") as prefix:
            values = [prefix["audit"][prefix["active"]]]
        if intervention > 40:
            with np.load(root / f"warmup_seed{seed}.npz") as warmup:
                values.append(warmup["audit"][warmup["active"]])
        prefixes[seed] = np.concatenate(values)
    for row in rows:
        assert row["checkpoint_sha256"] == protocol["teacher_checkpoint_sha256"]
        assert digest(root / row["trace_file"]) == row["trace_sha256"]
        assert row["max_qpos_error"] < 3e-6 and row["max_qvel_error"] < 1e-4
        assert row["released_rope_force_n"] == row["residual_rms_rad"] == 0
        if row["success30"]:
            assert row["hold_sec"] >= 30 and row["final_valid"] and not row["forbidden"]
            assert row["initial_penetration_m"] <= .001 and row["max_released_penetration_m"] <= .003
        with np.load(root / row["trace_file"]) as trace:
            assert trace["active"].shape == (protocol["suffix_steps"],)
            assert int(trace["active"].sum()) == row["native_steps"]
            assert bool(trace["success30"][-1]) == row["success30"]
            full = np.concatenate([prefixes[row["seed"]], trace["audit"][trace["active"]]])
        released = full[full[:, 10] > 0]
        row["full_path_max_released_depth_m"] = float(np.max(released[:, 6], initial=0))
        row["full_path_peak_force_n"] = float(np.max(full[:, 7], initial=0))
        row["full_path_forbidden"] = bool(np.any(released[:, 5] > 0))
        assert np.max(released[:, 8], initial=0) == 0
        if row["success30"]:
            assert row["full_path_max_released_depth_m"] <= .003 and not row["full_path_forbidden"]
    for seed in protocol["seeds"]:
        with np.load(root / f"seed{seed}_none_feedback.npz") as a, np.load(root / f"seed{seed}_none_replay.npz") as b:
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
    pairs = paired_rows(rows)
    overview = []
    for seed in protocol["seeds"]:
        part = [r for r in rows if r["seed"] == seed]
        for replay in (False, True):
            selected = [r for r in part if r["replay"] == replay]
            overview.append(dict(seed=seed, replay=replay, total=7,
                successes=sum(r["success30"] for r in selected),
                min_hold_sec=min(r["hold_sec"] for r in selected),
                max_depth_mm=1000 * max(r["full_path_max_released_depth_m"] for r in selected),
                peak_force_n=max(r["full_path_peak_force_n"] for r in selected)))
    full = [r for r in pairs if r["nominal_full_horizon"]]
    feedback_gain = sum(r["feedback_success"] and not r["replay_success"] for r in full)
    feedback_loss = sum(not r["feedback_success"] and r["replay_success"] for r in full)
    out = root / "report"
    out.mkdir(exist_ok=True)
    for name, data in (("overview.csv", overview), ("pairs.csv", pairs), ("episodes.csv", rows)):
        write_csv(out / name, data)
    evidence = dict(status="verified_training_diagnostic", full_nominal_pairs=len(full),
        feedback_gained=feedback_gain, feedback_lost=feedback_loss,
        results_sha256=digest(root / "results.json"), protocol_sha256=digest(root / "protocol.json"),
        summary_sha256=digest(root / "summary.json"), source_sha256=digest(Path(__file__)))
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    lines = ["# 固定补救老师的扰动邻域", "",
        f"同两个已见训练前缀，在{intervention * .05:.1f}秒施加0和xyz正负0.02m/s速度增量；总回合仍到40秒。",
        "老师继承1310720在线步和既有开发集专家选择。本项更新0，不是新视频或学生改进。", "",
        "| 训练种子 | 输入 | 成功/7 | 最短合格秒 | 最大释放后穿透mm | 峰值N |",
        "|---:|---|---:|---:|---:|---:|"]
    for r in overview:
        lines.append(f"| {r['seed']} | {'名义观测回放' if r['replay'] else '真实反馈'} | {r['successes']} | "
                     f"{r['min_hold_sec']:.2f} | {r['max_depth_mm']:.3f} | {r['peak_force_n']:.3f} |")
    lines += ["", f"名义完整的{len(full)}个非零扰动对：反馈相对回放新增{feedback_gain}、丢失{feedback_loss}。",
        "0扰动两输入全部字段相同，28后缀全部原生物理核验。失败案例仍保留。",
        "duration_sec是扰动后的后缀时长，不能与从0秒计时的D49直接比较。",
        "本表接触极值包括老师前缀、已学习策略热身和后缀；episodes.csv另保留后缀与full_path字段，不能用较短后缀掩盖早期接触。",
        "小范围扰动下成功不等于靠反馈救回；回放同样成功时不能宣称反馈增益。",
        "2秒干预时尚有绳索，不可当撤绳后滑落恢复；4秒干预才在无绳阶段，但仍只代表该小扰动范围。",
        "两训练状态不是独立保留测试，多个扰动也不是多个训练种子。未生成新训练集或新学生。"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(overview=overview, evidence=evidence)), flush=True)


if __name__ == "__main__":
    main()
