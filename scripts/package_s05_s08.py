"""Build a portable reporting archive with raw data, documents and videos, no weights."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/"deliverables/汇报资料_S05-S08_2026-10-12"
EXTENSIONS={".md",".csv",".json",".jsonl",".npz",".png",".jpg",".jpeg",".svg",".pdf",".mp4",".log",".txt",".py",".xml",".docx"}
DATA=ROOT/"artifacts/orca_benchmark_suite"
LINK=re.compile(r"(!?)\[([^\]\n]*)\]\(([^)\n]+)\)")
DIRECTORIES={}


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    mapping={}
    def add_tree(source,destination):
        if not source.is_dir():
            raise FileNotFoundError(source)
        DIRECTORIES[source.resolve()]=BUNDLE/destination
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS and not any(x in path.parts for x in ("__pycache__",".git","jax_cache")):
                if path.suffix==".mp4" and source.is_relative_to(DATA):
                    continue
                mapping[path.resolve()]=BUNDLE/destination/path.relative_to(source)
    for study in ("S05","S06","S07","S08"):
        add_tree(ROOT/"docs/experiments"/study,Path(study)/"文档")
        add_tree(DATA/study,Path(study)/"数据")
        for video in sorted((ROOT/"deliverables").glob(study+"_*")):
            if video.is_dir():
                add_tree(video,Path(study)/"视频"/video.name)
        for path in (ROOT/"scripts").glob("*.py"):
            if study.lower() in path.stem:
                mapping[path.resolve()]=BUNDLE/study/"源码"/path.name
    for folder in sorted(DATA.glob("S08_wrap_*")):
        add_tree(folder,Path("S08/摆位与几何开发")/folder.name)
    add_tree(ROOT/"artifacts/orca_collision_refit/O01_hic53_source_object",Path("S05/来源证据/物体与跟踪对应"))
    for source,name in (
        ("artifacts/cube20/hic20/prepared/human_cube.npz","human_cube.npz"),
        ("artifacts/cube20/hic20/dexmv_retarget/adroit_cube.npz","adroit_cube.npz"),
        ("artifacts/cube20/hic20/videos/20/input_rgb.mp4","HIC20_原始RGB_非策略仿真.mp4")):
        mapping[(ROOT/source).resolve()]=BUNDLE/"S05/来源证据"/name
    add_tree(ROOT/"artifacts/orca_cylinder_s02/source/selection96",Path("S06/来源证据/selection96"))
    add_tree(ROOT/"docs/literature",Path("文献综述"))
    for name in ("orca_s05_env.py","orca_s06_env.py","orca_s07_env.py","orca_s08_env.py",
                 "orca_residual_controls.py","orca_residual_env.py","orca_native_physics.py",
                 "orca_suite_env.py","train_orca_matched.py","train_orca_comparison.py",
                 "s05_pad_contacts.py","probe_s08_wrap.py","render_s05_clips.py","package_s05_s08.py"):
        path=ROOT/"scripts"/name
        mapping.setdefault(path.resolve(),BUNDLE/"公共源码"/name)
    # Preserve already-reviewed background, not obsolete historical result tables.
    for name in ("00_对话总结.md","01_综述格式待修清单.md","02_汇报主线与方法原理.md"):
        path=ROOT/"deliverables/汇报资料_2026-10-10"/name
        if path.exists():
            mapping[path.resolve()]=BUNDLE/"背景说明"/name
    assert len(set(mapping.values()))==len(mapping)
    return mapping


def rewrite_markdown(text,source,target,mapping,external):
    def replace(match):
        image,label,link=match.groups()
        if re.match(r"^[a-zA-Z][\w+.-]*:",link) or link.startswith("#"):
            return match.group(0)
        raw=link.strip("<>")
        pathpart,sep,anchor=raw.partition("#")
        path=(source.parent/pathpart).resolve()
        destination=mapping.get(path)
        if destination is None and path.is_dir():
            parents=[a for a in DIRECTORIES if path.is_relative_to(a)]
            if parents:
                parent=max(parents,key=lambda a:len(a.parts))
                destination=DIRECTORIES[parent]/path.relative_to(parent)
        if destination is not None:
            rel=os.path.relpath(destination,target.parent)
            return f"{image}[{label}]({rel}{sep}{anchor})"
        external.append(dict(document=str(target.relative_to(BUNDLE)),label=label,original=link,
                             resolved_source=str(path)))
        return f"{label}（原工程引用：`{link}`，本包未附）"
    return LINK.sub(replace,text)


def check_refresh_targets():
    # Refuse to overwrite reader edits in an already-delivered bundle.
    with (BUNDLE/"文件来源与校验.csv").open() as f:
        for row in csv.DictReader(f):
            path=BUNDLE/row["file"]
            if digest(path)!=row["package_sha256"]:
                raise ValueError("Bundle file changed locally: "+str(path))


def build(mapping,refresh=False):
    check=json.loads((DATA/"S08/report/verification.json").read_text())
    assert check["status"]=="verified" and check["total_test_episodes"]==3424
    assert (ROOT/"docs/experiments/S08/完整对比结果.md").exists()
    assert (ROOT/"deliverables/S08_纵向包覆完整对比_2026-10-12/来源与校验.json").exists()
    archive=BUNDLE.with_suffix(".zip")
    temporary=archive.with_suffix(".zip.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    if refresh:
        check_refresh_targets()
    elif archive.exists():
        raise FileExistsError(archive)
    BUNDLE.mkdir(exist_ok=refresh)
    manifest,external=[],[]
    for source,target in mapping.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.suffix==".md":
            target.write_text(rewrite_markdown(source.read_text(),source,target,mapping,external))
        else:
            shutil.copy2(source,target)
        manifest.append(dict(source=str(source.relative_to(ROOT)),file=str(target.relative_to(BUNDLE)),
            bytes=target.stat().st_size,source_sha256=digest(source),package_sha256=digest(target)))
    with (BUNDLE/"文件来源与校验.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    (BUNDLE/"未附的历史外部引用.json").write_text(json.dumps(external,ensure_ascii=False,indent=2)+"\n")
    descriptions={"S05":"方块末节指腹修复，单训练种子的小规模独立验证。",
                  "S06":"圆柱三指重定向与校正开发，单训练种子可行性验证。",
                  "S07":"圆柱三指完整复验：3种子、扰动、参考/残差消融。",
                  "S08":"纵向圆柱掌面包覆：摆位修复后重新采集示范、训练和完整比较。"}
    for study,description in descriptions.items():
        folder=BUNDLE/study
        lines=[f"# {study}目录","",description,"",
            "## 先读说明","","- [文档入口](文档/README.md)",
            "- [全部原始数据](数据/)","- [主要视频](视频/)","- [核心源码](源码/)","",
            "## 文档"," "]
        lines += [f"- [{p.name}](文档/{p.name})" for p in sorted((folder/"文档").glob("*.md"))]
        lines += ["","## 视频分组",""]
        lines += [f"- [{p.parent.name}]({p.relative_to(folder).as_posix()})" for p in sorted((folder/"视频").rglob("目录.md"))]
        lines += ["","## 原始数据说明","",
            "CSV/JSON保存训练曲线、逐回合结果、配置与选择规则；NPZ保存参考、示范和评估轨迹，不是神经网络权重。原始日志及失败开发候选保留，不能把每个P开头目录当成一次PPO训练。",
            "","历史JSON和日志里的绝对路径保留原始内容以便核对来源，并不表示移动后能直接运行。Markdown可附的链接已改成本包相对路径；未附的早期历史引用明确标出。",
            "","本目录不含PKL权重、完整手部STL资产或conda环境，所以是汇报/分析资料，不是独立可运行的训练工程。原工程未删除或移动。"]
        if study=="S08":
            lines += ["","- [新完整对比结果](文档/完整对比结果.md)","- [摆位与几何开发原始资料](摆位与几何开发/)"]
        if study in ("S05","S06"):
            lines += ["","- [示范来源证据](来源证据/)"]
        if study=="S07":
            lines += ["","- [与S06共用的处理后源片段](../S06/来源证据/selection96/)"]
        (folder/"目录.md").write_text("\n".join(lines)+"\n")
    counts=Counter(row["file"].split("/")[0] for row in manifest)
    sizes={key:sum(row["bytes"] for row in manifest if row["file"].split("/")[0]==key) for key in counts}
    lines=["# S05至S08汇报资料总目录","",
        "这是数据、思路、方案、原理、分析和主要展示视频的资料包，不含策略权重和旧checkpoint。原工程保持原位；可整体移动本目录或同名ZIP。",
        "","- [先看：历次实验汇报主线与数据总览](S08/文档/历次实验汇报主线.md)",
        "- [新实验：S08完整方法对比结果](S08/文档/完整对比结果.md)",
        "- [NPZ、CSV、日志分别是什么，如何读数据](S08/文档/资料包数据阅读说明.md)",
        "- [逆运动学是什么、具体用了什么算法](S08/文档/逆运动学与参考生成原理.md)",
        "","## 建议阅读顺序","",
        "1. 先看各实验目录与结果说明，区分开发验证和完整对比。",
        "2. 看S08新实验的三方法原理、正常/扰动成功率和训练曲线。",
        "3. 用逐回合数据、冻结选模规则和失败视频核对结论。",
        "4. 文献综述作为研究背景；文档中的未来工作不能冒充已完成实验。",
        "","| 实验 | 内容 | 入口 |","|---|---|---|"]
    lines += [f"| {s} | {description} | [目录]({s}/目录.md) |" for s,description in descriptions.items()]
    lines += ["","- [文献综述](文献综述/)","- [方法背景与综述格式说明](背景说明/)",
        "- [共享控制/训练源码](公共源码/)","- [完整文件来源与SHA256](文件来源与校验.csv)",
        "- [未附的历史外部引用](未附的历史外部引用.json)","",
        "- [文件与目录交付校验](交付校验.json)","- [全部视频解码校验](视频校验.json)","",
        "## 解读边界","",
        "`背景说明/00_对话总结.md`是截至2026-10-10的历史总结，用于回顾研究目标，不是S05至S08本轮结果。当前数字以各实验原始数据和新总览为准。",
        "",
        "S05方块、S06/S07三指圆柱、S08掌面包覆的成功条件不同，不能把百分比串成同一个排行榜。S08用几何参考，不是新的RGB视频提取。S07/S08各3个训练种子也不能代表穷尽调参后的算法极限。",
        "","主要策略视频只保留手心视角，外观和碰撞几何两版；展示回合有选择，性能结论以完整逐回合统计为准。几何开发视频明确标记非PPO。另附一段S05的HIC20原始RGB作为来源证据，不是策略执行视频；S06/S07只附实际使用的处理后片段，不伪造尚未核实的RGB对应。",
        "","## 大小与文件组成","","| 分组 | 文件数 | 大小MiB |","|---|---:|---:|"]
    lines += [f"| {key} | {counts[key]} | {sizes[key]/1024**2:.2f} |" for key in counts]
    lines += ["","较大的NPZ通常是多回合、多时刻的状态/接触轨迹，而不是模型权重。权重PKL全部排除。文档中提及的权重路径用于来源追溯，文件不在本包。"]
    (BUNDLE/"目录.md").write_text("\n".join(lines)+"\n")
    (BUNDLE/"README.md").write_text("# S05至S08汇报资料\n\n从[总目录](目录.md)开始阅读。\n")
    media=[]
    for path in sorted(BUNDLE.rglob("*.mp4")):
        info=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames",
            "-show_entries","stream=codec_name,width,height,nb_read_frames,avg_frame_rate",
            "-show_entries","format=duration","-of","json",str(path)],text=True))
        assert float(info["format"]["duration"])>0
        assert any(int(s.get("nb_read_frames",0))>0 for s in info["streams"])
        subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-f","null","-"],
            check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        media.append(dict(file=str(path.relative_to(BUNDLE)),sha256=digest(path),decoded=True,probe=info))
    (BUNDLE/"视频校验.json").write_text(json.dumps(media,ensure_ascii=False,indent=2)+"\n")
    audit=dict(status="pending",copied_source_files=len(manifest),
        copied_source_hashes_match=True,package_hashes_match=True,local_markdown_links_valid=True,
        weights_pkl=0,video_files=len(media),all_videos_decoded=True,
        studies=["S05","S06","S07","S08"],
        s08=json.loads((DATA/"S08/report/verification.json").read_text()),
        archive_validation="ZIP CRC and every entry SHA256 are checked after creation; see sibling delivery JSON.")
    (BUNDLE/"交付校验.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    assert verify_directory()==len(manifest)
    audit["status"]="verified"
    (BUNDLE/"交付校验.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    with zipfile.ZipFile(temporary,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                z.write(path,str(path.relative_to(BUNDLE.parent)))
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        files=[p for p in BUNDLE.rglob("*") if p.is_file()]
        assert len(z.namelist())==len(files)
        for path in files:
            assert hashlib.sha256(z.read(str(path.relative_to(BUNDLE.parent)))).hexdigest()==digest(path)
    # Keep the previous ZIP intact until the replacement passes all checks.
    temporary.replace(archive)
    final=dict(status="verified",directory=str(BUNDLE),archive=str(archive),files=len(files),
        archive_bytes=archive.stat().st_size,archive_sha256=digest(archive),
        full_zip_crc_and_sha256_checked=True,**{k:v for k,v in audit.items() if k!="status"})
    archive.with_suffix(".zip.sha256").write_text(f"{final['archive_sha256']}  {archive.name}\n")
    archive.with_suffix(".交付校验.json").write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(final,ensure_ascii=False),flush=True)


def verify_directory():
    assert not list(BUNDLE.rglob("*.pkl"))
    with (BUNDLE/"文件来源与校验.csv").open() as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        assert digest(BUNDLE/r["file"])==r["package_sha256"],r["file"]
        assert digest(ROOT/r["source"])==r["source_sha256"],r["source"]
    broken=[]
    for path in BUNDLE.rglob("*.md"):
        for _,label,link in LINK.findall(path.read_text()):
            if re.match(r"^[a-zA-Z][\w+.-]*:",link) or link.startswith("#"):
                continue
            target=(path.parent/link.strip("<>").partition("#")[0]).resolve()
            if not target.is_relative_to(BUNDLE) or not target.exists():
                broken.append((str(path),link))
    assert not broken,broken[:30]
    return len(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("inventory","build","refresh","verify"))
    args=p.parse_args()
    if args.phase=="verify":
        print("VERIFIED",verify_directory());return
    mapping=inventory()
    if args.phase=="inventory":
        groups=Counter()
        for source,target in mapping.items():
            groups[str(target.relative_to(BUNDLE)).split("/")[0]]+=source.stat().st_size
        print(json.dumps(dict(files=len(mapping),MiB={k:v/1024**2 for k,v in groups.items()}),ensure_ascii=False,indent=2))
    else:
        build(mapping,refresh=args.phase=="refresh")


if __name__=="__main__":
    main()
