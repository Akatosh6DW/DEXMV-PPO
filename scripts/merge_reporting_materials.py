"""Merge existing study evidence into one directory; never create a ZIP or copy weights."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import package_s05_s08 as shared

ROOT = shared.ROOT
OLD = ROOT / "deliverables/汇报资料_S05-S08_2026-10-12"
OUT = ROOT / "deliverables/完整实验汇报资料_2026-10-12"
EXT = shared.EXTENSIONS | {".tex", ".bib", ".bst", ".diff", ".npy", ".yaml", ".yml", ".sh", ".gif"}
EXCLUDED = {".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".zip", ".stl", ".obj", ".msh"}
STUDIES = {
    "S02_方块": ("artifacts/orca_benchmark_suite/S02", "docs/experiments/S02", 77600),
    "S03_调优与训练来源": ("artifacts/orca_benchmark_suite/S03", "docs/experiments/S03", 91800),
    "S04_方块": ("artifacts/orca_benchmark_suite/S04", "docs/experiments/S04", 4100),
    "S02_圆柱": ("artifacts/orca_cylinder_s02", None, 77600),
}
LEGACY_PACKS = (ROOT / "deliverables/汇报资料_2026-10-10",
                ROOT / "deliverables/Orca_S02_S04_Experiments")


def read_manifest(base):
    with (base / "文件来源与校验.csv").open() as f:
        return list(csv.DictReader(f))


def link_candidates(source, raw):
    original = (source.parent / raw).resolve()
    candidates = [original]
    for pack in LEGACY_PACKS:
        if original.is_relative_to(pack):
            rel = original.relative_to(pack)
            candidates.append(ROOT / rel)
            if rel.parts and rel.parts[0] in ("S02", "S04"):
                candidates.append(shared.DATA / rel)
            if rel.parts and rel.parts[0] == "S04_training_sources":
                candidates.append(shared.DATA / "S03/confirmation" / Path(*rel.parts[1:]))
    return candidates


def supplement_links(mapping):
    # Follow individual local attachments, never expand linked directories or weights.
    pending = [p for p in mapping if p.suffix == ".md"]
    visited = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        for match in shared.LINK.finditer(source.read_text()):
            link = match.group(3)
            if re.match(r"^[a-zA-Z][\w+.-]*:", link) or link.startswith("#"):
                continue
            candidates = link_candidates(source, link.strip("<>").partition("#")[0])
            if any(p in mapping for p in candidates):
                continue
            for path in candidates:
                if (not path.is_relative_to(ROOT) or path.is_relative_to(OUT)
                        or not path.is_file() or path.suffix.lower() not in EXT
                        or path.suffix.lower() in EXCLUDED
                        or {".git", ".env", "__pycache__"}.intersection(path.parts)):
                    continue
                mapping[path] = OUT / "历史引用补充" / path.relative_to(ROOT)
                if path.suffix == ".md":
                    pending.append(path)
                break


def inventory():
    base = OUT if OUT.exists() else OLD
    rows = read_manifest(base)
    mapping = {(ROOT / r["source"]).resolve(): OUT / r["file"] for r in rows}
    directories = {}

    def add_tree(rel, destination):
        source = ROOT / rel
        if not source.is_dir():
            raise FileNotFoundError(source)
        directories[source.resolve()] = OUT / destination
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXT:
                continue
            if {".git", "__pycache__", "jax_cache"}.intersection(path.parts):
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(ROOT):
                raise ValueError("External source: " + str(path))
            mapping.setdefault(resolved, OUT / destination / path.relative_to(source))

    for group, (data, docs, _) in STUDIES.items():
        add_tree(data, Path(group) / "数据")
        if docs:
            add_tree(docs, Path(group) / "文档")
    for s in ("S05", "S06", "S07", "S08"):
        directories[ROOT / "docs/experiments" / s] = OUT / s / "文档"
        directories[shared.DATA / s] = OUT / s / "数据"
    add_tree("docs/experiments/Orca_Video_PPO", Path("S02_S04综合讲解/总报告与注释源码"))
    add_tree("docs/experiments/S02_S04_comparison", Path("S02_S04综合讲解/两轮对比"))
    add_tree("artifacts/orca_benchmark_suite/S02_S04_comparison", Path("S02_S04综合讲解/对比原始数据"))
    add_tree("docs/experiments/merged_reporting", Path("合并说明"))
    add_tree("deliverables/抓取展示视频_2026-10-12", Path("主要短视频"))
    add_tree("artifacts/cube20/hic20", Path("共同来源/HIC20"))
    for name in ("D01_quality_demos", "E01_flex50_refinement"):
        add_tree("artifacts/orca_cube_acquisition/" + name, Path("共同来源") / name)
    add_tree("deliverables/Orca_Video_PPO_S04/reference", Path("共同来源/校正参考"))
    add_tree("docs/literature", Path("文献综述"))
    add_tree("scripts", Path("公共源码/其余处理与分析脚本"))
    for path in (ROOT / "docs/experiments").glob("cylinder*.md"):
        mapping.setdefault(path.resolve(), OUT / "S02_圆柱/文档" / path.name)
    for name in ("README.md", "03_实验全解.md", "04_实验完整数据附录.md"):
        path = LEGACY_PACKS[0] / name
        mapping.setdefault(path.resolve(), OUT / "背景说明" / ("旧汇报目录.md" if name == "README.md" else name))
    path = LEGACY_PACKS[1] / "EXPERIMENTS_EXPLAINED.md"
    mapping[path.resolve()] = OUT / "S02_S04综合讲解/两轮实验详细区别.md"
    for path in (ROOT / "docs").glob("*.md"):
        if any(k in path.stem for k in ("cube", "benchmark", "residual", "matched", "guidance", "experiment")) and "cap" not in path.stem:
            mapping.setdefault(path.resolve(), OUT / "背景说明/早期链路与诊断" / path.name)
    for name in ("README.md", "docs/demo_gen.md", "docs/env.md"):
        path = ROOT / "DexMV" / name
        if path.exists():
            mapping.setdefault(path.resolve(), OUT / "共同来源/DexMV说明" / Path(name).name)
    # Frozen protocols may reference shared inputs outside their own study tree.
    for rel, _, _ in STUDIES.values():
        for path in (ROOT / rel).glob("*protocol.json"):
            for name in json.loads(path.read_text()).get("source_hashes", {}):
                src = (ROOT / name).resolve()
                if src.is_file() and src.is_relative_to(ROOT) and src.suffix in EXT:
                    mapping.setdefault(src, OUT / "共同来源/协议引用" / src.relative_to(ROOT))
    supplement_links(mapping)
    # Recover directory destinations for prior, already-portable S05-S08 links.
    for source, target in mapping.items():
        directories.setdefault(source.parent, target.parent)
    assert len(mapping) == len(set(mapping.values())), "Conflicting target files"
    assert not any(p.suffix.lower() in EXCLUDED for p in mapping)
    return mapping, directories, rows


def rewrite(text, source, target, mapping, directories, missing):
    def replace(match):
        image, label, link = match.groups()
        if re.match(r"^[a-zA-Z][\w+.-]*:", link) or link.startswith("#"):
            return match.group(0)
        raw, sep, anchor = link.strip("<>").partition("#")
        candidates = link_candidates(source, raw)
        original = candidates[0]
        for candidate in candidates:
            destination = mapping.get(candidate)
            if destination is None and candidate.is_dir():
                parents = [p for p in directories if candidate.is_relative_to(p)]
                if parents:
                    parent = max(parents, key=lambda p: len(p.parts))
                    proposed = directories[parent] / candidate.relative_to(parent)
                    if any(t.is_relative_to(proposed) for t in mapping.values()):
                        destination = proposed
            if destination is not None:
                rel = os.path.relpath(destination, target.parent)
                return f"{image}[{label}]({rel}{sep}{anchor})"
        missing.append(dict(document=str(target.relative_to(OUT)), label=label,
                            original=link, resolved_source=str(original)))
        return f"{label}（原工程引用：`{link}`，本包未附）"
    return shared.LINK.sub(replace, text)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def indices(mapping, manifest):
    episode_counts = {}
    for group, (source, docs, expected) in STUDIES.items():
        folder = OUT / group
        with (folder / "数据/report/episodes.csv").open() as f:
            count = sum(1 for _ in csv.DictReader(f))
        assert count == expected, (group, count)
        episode_counts[group] = count
        lines = [f"# {group}目录", "", f"已有逐回合汇总记录：{count}行。"]
        if group.startswith("S03"):
            lines += ["", "**扩展评估未全部完成。** 此处保留已有开发、训练与测试，不能写成完整矩阵。"]
        if group.startswith("S04"):
            lines += ["", "本轮没有新训练；使用S03三个组各10个种子的163840步检查点，不是655360步最终权重。",
                      "", "- [30次训练来源索引](训练来源索引.csv)",
                      "- [S03完整已有训练记录](../S03_调优与训练来源/数据/confirmation/)"]
        lines += ["", "## 文档", ""]
        lines += [f"- [{p.name}]({p.relative_to(folder).as_posix()})" for p in sorted((folder / "文档").glob("*.md"))]
        lines += ["", "## 数据与视频", "", "- [完整原始数据目录](数据/)",
                  "- [逐回合CSV](数据/report/episodes.csv)", "- [统计、图表与分析](数据/report/)",
                  "- [主要手心短视频](../主要短视频/目录.md)"]
        video_index = folder / "数据/videos/index.md"
        if video_index.exists():
            lines += ["- [本轮原始实验视频索引](数据/videos/index.md)"]
        lines += ["", "资料按原文保留，Markdown链接改为包内相对路径。PKL和完整网格不附；配置中的权重路径仅作追溯。",
                  "本次没有新训练或新评估，复制与校验不等于重新完成物理实验。"]
        (folder / "目录.md").write_text("\n".join(lines) + "\n")
    protocol = json.loads((shared.DATA / "S04/protocol.json").read_text())
    sources = []
    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("checkpoint"), str):
                p = (ROOT / value["checkpoint"]).resolve()
                if "step_000163840" in p.parts:
                    metrics = p.parent.parent / "metrics.csv"
                    assert metrics in mapping, metrics
                    sources.append(dict(checkpoint_source=str(p.relative_to(ROOT)),
                        checkpoint_sha256=shared.digest(p), weights_included=False,
                        metrics_in_bundle=str(mapping[metrics].relative_to(OUT))))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(protocol)
    sources = list({r["checkpoint_source"]: r for r in sources}.values())
    assert len(sources) == 30, len(sources)
    with (OUT / "S04_方块/训练来源索引.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(sources[0])); w.writeheader(); w.writerows(sources)
    groups = Counter(r["file"].split("/")[0] for r in manifest)
    labels = {
        "S02_方块": "旧判据下完整多条件与消融，77600回合",
        "S03_调优与训练来源": "已有开发和长训练，扩展评估未全部完成",
        "S04_方块": "163840步有限预算复测，4100回合，复用S03权重",
        "S02_圆柱": "换圆柱和DexMV示范后重新训练，77600回合",
        "S05": "方块指腹修复", "S06": "圆柱三指重定向开发",
        "S07": "圆柱三指完整复验", "S08": "纵向包覆几何参考完整比较",
    }
    lines = ["# 完整实验汇报资料总目录", "", "当前实体目录包含方块S02/S03/S04、圆柱S02及S05至S08。",
        "原工程保留；没有权重PKL、完整网格和conda环境。本次只合并文件夹，不生成或更新ZIP。",
        "旧S05-S08目录名是兼容入口，移动时请移动本实体目录；旧ZIP不含本次合并内容。", "",
        "## 先看这些", "", "- [合并说明与阅读顺序](合并说明/合并说明与阅读顺序.md)",
        "- [S02/S04完整故事、原理与结论](S02_S04综合讲解/总报告与注释源码/report.md)",
        "- [S02和S04的任务、配方、统计与区别](S02_S04综合讲解/两轮实验详细区别.md)",
        "- [从步数、奖励、BC到PPO的实验全解](背景说明/03_实验全解.md)",
        "- [历史完整数据附录](背景说明/04_实验完整数据附录.md)",
        "- [S05至S08汇报主线](S08/文档/历次实验汇报主线.md)",
        "- [IK、BC、DexMV与残差的关系](S08/文档/逆运动学与参考生成原理.md)",
        "- [主要短视频：方块S02、方块S04、圆柱S02](主要短视频/目录.md)",
        "", "## 实验入口", "", "| 实验 | 内容 | 入口 |", "|---|---|---|"]
    lines += [f"| {k} | {v} | [目录]({k}/目录.md) |" for k, v in labels.items()]
    lines += ["", "## 资料与核验", "", "- [文献综述](文献综述/)",
        "- [HIC、BC示范、校正与协议共享来源](共同来源/)",
        "- [公共源码](公共源码/)", "- [文件来源与哈希](文件来源与校验.csv)",
        "- [历史引用补充：旧说明链接到的附件](历史引用补充/)",
        "- [全部视频索引](全部视频索引.md)", "- [视频解码校验](视频校验.json)",
        "- [交付校验](交付校验.json)", "- [仍未附的历史引用](未附的历史外部引用.json)",
        "", "## 不能混读的地方", "",
        "S02/S04旧接触判据与后来的指腹/包覆判据不同，不按所有成功率统一排名。S04有意针对有限预算且复用S03权重，不是全新训练重复。S03未完成部分不补造数据。",
        "S08仍为几何参考；新DexMV包覆实验尚未完成。视频是固定案例展示，不代表总体成功率；原始RGB与仿真视频分开。",
        "历史文档中的旧进度、文件路径和归档说明保留作历史记录，当前收录范围以本页为准。",
        "历史引用补充只收录被文档引用的现有文件，不代表所有旧实验都已完整归档；其中旧校验报告仅反映当时的归档状态。",
        "", "## 文件组成", "", "| 分组 | 来源文件数 | MiB |", "|---|---:|---:|"]
    lines += [f"| {k} | {n} | {sum(r['bytes'] for r in manifest if r['file'].split('/')[0]==k)/2**20:.2f} |"
              for k, n in sorted(groups.items())]
    (OUT / "目录.md").write_text("\n".join(lines) + "\n")
    (OUT / "README.md").write_text("# 完整实验汇报资料\n\n从[总目录](目录.md)开始。只更新文件夹，不更新ZIP；不含权重。\n")
    return episode_counts


def build():
    mapping, directories, old_rows = inventory()
    base = OUT if OUT.exists() else OLD
    for r in old_rows:
        assert shared.digest(base / r["file"]) == r["package_sha256"], "Locally edited delivery: " + r["file"]
    new_bytes = sum(p.stat().st_size for p, t in mapping.items() if not (base / t.relative_to(OUT)).exists())
    assert shutil.disk_usage(ROOT).free > new_bytes + 1024**3, "Insufficient disk reserve"
    old_zip = OLD.with_suffix(".zip")
    old_zip_hash = shared.digest(old_zip)
    prior_media = json.loads((base / "视频校验.json").read_text())
    if not OUT.exists():
        OLD.rename(OUT)
        OLD.symlink_to(OUT.name, target_is_directory=True)
    write_json(OUT / "交付校验.json", dict(status="merging", archive_updated=False))
    manifest, missing = [], []
    for i, (source, target) in enumerate(sorted(mapping.items()), 1):
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = shared.digest(source)
        if source.suffix == ".md":
            target.write_text(rewrite(source.read_text(), source, target, mapping, directories, missing))
        elif not target.exists() or shared.digest(target) != checksum:
            shutil.copy2(source, target)
        manifest.append(dict(source=str(source.relative_to(ROOT)), file=str(target.relative_to(OUT)),
            bytes=target.stat().st_size, source_sha256=checksum, package_sha256=shared.digest(target)))
        if i % 5000 == 0:
            print(json.dumps(dict(copied=i, total=len(mapping))), flush=True)
    with (OUT / "文件来源与校验.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
    write_json(OUT / "未附的历史外部引用.json", missing)
    counts = indices(mapping, manifest)
    cached = {r["sha256"]: r for r in prior_media if r["decoded"]}
    media = []
    for path in sorted(OUT.rglob("*.mp4")):
        checksum = shared.digest(path)
        if checksum in cached:
            item = dict(cached[checksum], file=str(path.relative_to(OUT)), validation="matching previously decoded bytes")
        else:
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
                "-show_entries", "stream=codec_name,width,height,nb_read_frames", "-show_entries",
                "format=duration", "-of", "json", str(path)], text=True))
            assert float(info["format"]["duration"]) > 0
            assert any(int(s.get("nb_read_frames", 0)) > 0 for s in info["streams"])
            subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-f", "null", "-"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            item = dict(file=str(path.relative_to(OUT)), sha256=checksum, decoded=True, probe=info,
                        validation="full decode during merge")
            cached[checksum] = item
        media.append(item)
        if len(media) % 20 == 0:
            print("VIDEOS", len(media), flush=True)
    write_json(OUT / "视频校验.json", media)
    lines = ["# 全部视频索引", "", "主要短片优先看[18段集中展示](主要短视频/目录.md)。下列同时保留原始实验视频，可能与短片内容重叠。",
        "视频是案例，不是独立训练；原始RGB、Adroit重建、几何开发和策略仿真按所在目录区分。", "",
        "| 文件 | 秒 |", "|---|---:|"]
    lines += [f"| [{r['file']}]({r['file']}) | {float(r['probe']['format']['duration']):.2f} |" for r in media]
    (OUT / "全部视频索引.md").write_text("\n".join(lines) + "\n")
    shared.BUNDLE = OUT
    assert shared.verify_directory() == len(manifest)
    assert not any(p.suffix.lower() in EXCLUDED for p in OUT.rglob("*") if p.is_file())
    assert shared.digest(old_zip) == old_zip_hash
    audit = dict(status="verified", directory=str(OUT), legacy_entry=str(OLD),
        copied_source_files=len(manifest), copied_bytes=sum(r["bytes"] for r in manifest),
        copied_source_hashes_match=True, package_hashes_match=True, local_markdown_links_valid=True,
        episode_rows=counts, s04_training_sources=30, incomplete_studies=["S03"],
        studies=list(STUDIES) + ["S05", "S06", "S07", "S08"], video_files=len(media), all_videos_decoded=True,
        weights_pkl=0, archive_updated=False, previous_zip_sha256=old_zip_hash,
        archive_matches_current_directory=False, original_results_modified=False,
        omitted_historical_links=len(missing),
        note="Merge/source-copy/media verification only. Historical experiment audits retained, no retraining or new performance claim.")
    write_json(OUT / "交付校验.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("inventory", "build", "verify"))
    phase = parser.parse_args().phase
    if phase == "build":
        build()
    elif phase == "verify":
        shared.BUNDLE = OUT
        print("VERIFIED", shared.verify_directory())
    else:
        mapping, _, _ = inventory()
        print(json.dumps(dict(files=len(mapping), total_MiB=sum(p.stat().st_size for p in mapping)/2**20), indent=2))


if __name__ == "__main__":
    main()
