"""Build a read-only study package: annotated source snapshots and verified data."""

import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tokenize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/Orca_Video_PPO"
BASE = ROOT / "artifacts/orca_benchmark_suite"

# Each anchor must identify exactly one location. Only comments are inserted.
NOTES = {
    "retarget_hic_cube_to_dexmv.py": ("01 视频跟踪到DexMV中间手型", [
        ("def build_optimizer():", "建立DexMV的受关节限制位置优化器；这里做运动学拟合，不是TRPO或PPO训练。"),
        ("    objects = world_transform @ objects", "手和物体必须使用同一个坐标变换，否则指尖与物体会错位。"),
        ("    for i, desired in enumerate(optimization_target):", "上一帧关节解作为下一帧起点，逐帧求可实现的动作；拟合误差仍需另外检查。"),
        ("    np.savez_compressed(a.output / \"adroit_cube.npz\"", "文件保存关节、拟合点、目标点和物体姿态；后续Orca重定向用空间关系，不直接复制关节角。"),
    ]),
    "prepare_hic_orca_object.py": ("02 绑定同源物体及坐标", [
        ("def object_local(points, poses):", "把每帧手指位置转到该帧物体坐标系；移动相机不应改变抓在物体哪个面上。"),
        ("def build(source, model_path, output):", "绑定同一HIC序列的网格、跟踪和重定向产物，检查单位与尺寸。"),
    ]),
    "probe_hic53_orca.py": ("03 将物体相对轨迹拟合到Orca", [
        ("    selected = np.arange(10 if args.contact_fingers_only else 7, 17)", "O15实际命令只拟合拇指和食指的7个执行器，其他指不参与此阶段的IK。"),
        ("    def endpoint_residual(x):", "先同时寻找可达末姿态与物体刚体变换；误差含指尖距离、正则项和可选碰撞惩罚。"),
        ("    for frame in reversed(range(len(local))):", "从已选末姿态倒着解整段轨迹，用相邻解约束跳变；不是逐个关节一一对应。"),
        ("    times = np.arange(31) * .05", "选定视频阶段重采样为1.5秒、31个时刻；空间尺度仍是1，不缩小53mm物体。"),
    ]),
    "probe_orca_cube_acquisition.py": ("04 E01仿真校正参考", [
        ("    opened = np.arange(17)", "开发阶段打开部分屈曲关节，避免把完整抓牢的姿态当成起点；E01 opening=0.5。"),
        ("    for generation in range(args.generations):", "采样有界关节偏置，再保留高分候选更新采样分布。这是参考搜索，不是PPO。"),
        ("            controls = np.clip(reference +", "把小偏置在1.5秒内渐渐加到视频参考上；因此最终先验含仿真修正，不是原视频原样。"),
        ("    for seed in range(710, 718):", "独立开发初态检查参考；这些不是S02/S04最终测试初态。"),
    ]),
    "train_orca_acquisition.py": ("05 参考回放生成D01示范", [
        ("    if args.mode in (\"collect\", \"smoke\"):", "collect模式由参考控制器给动作，用仿真补齐视频没有的速度、触觉等观测。"),
        ("        if args.mode == \"collect\":", "仅保留成功且通过几何门槛的轨迹；每条都来自同一视频的扰动回放。"),
        ("            np.savez_compressed(path,", "BC文件保存obs/action/active，即状态、正确动作、有效时间步；不是视频像素。"),
    ]),
    "train_orca_matched.py": ("06 BC预训练及共享初始化", [
        ("def initialize_matched(env, seed, std):", "纯策略从起手目标初始化，残差从零修正初始化；两者的动作语义不同。"),
        ("def dataset_arrays(path, tactile, model_hash, reference_hash, physics_backend=", "加载前核对手模型、参考和物理后端的哈希，防止混用旧36mm任务的示范。"),
        ("    split = max(2, len(obs) * 3 // 4)", "按整条轨迹划分训练/验证，避免把同一轨迹相邻帧分到两边；仍不算跨视频验证。"),
        ("    def loss(policy, obs, actions):", "拟合策略均值输出与示范动作的反tanh值，因为网络动作输出随后经过tanh归一化。"),
        ("        i = jax.random.randint(a, (128,),", "每批各取128个接近阶段和保持阶段样本，防止长保持段淹没短合拢段。"),
    ]),
    "orca_residual_controls.py": ("07 残差合成公式", [
        ("def load_reference(path, low, high, initial, dt, steps, max_delta):", "检查时间及关节范围，将离线参考插值到控制时刻；末帧后保持末值。"),
        ("    for target in raw:", "预处理参考本身的变化速率；实际执行的合成目标还要经过环境的统一速率限制。"),
        ("def compose(base, action, bound, low, high, xp=np):", "真正的残差架构在此：参考角+有界修正。每步独立相加，不累计成无限漂移。"),
    ]),
    "orca_residual_env.py": ("08 把每步参考与策略接入环境", [
        ("    def next_base(self, step):", "以当前控制步查参考，而不是用实时RGB预测；超过轨迹尾部时保持。"),
        ("    def step(self, state, action):", "residual分支将17维动作解释为修正；pure分支将同样17维解释为完整关节目标。"),
        ("        effective = nxt.data.ctrl-base", "requested是策略请求值；effective是限幅/限速后实际控制值减参考，二者不能混用。"),
        ("    def _obs(self, data, info):", "残差策略看到归一化参考和时间；纯PPO的参考通道填零，维度保持一致。"),
    ]),
    "orca_benchmark_env.py": ("09 基础控制与有效抓握判据", [
        ("        ctrl = jp.clip(desired, state.data.ctrl - delta, state.data.ctrl + delta)", "所有路线每50ms最多改变0.1rad目标；这是目标变化限制，不是实际关节速度硬保证。"),
        ("        stable = (t >= 3)", "撤托后必须有拇指和另一指接触、物体未明显下降、无禁接触，才累计连续保持。"),
        ("        reward = (3 * proximity", "注意：这是父类历史奖励，当前链路会被OrcaRefitGrasp覆盖；不要据这一行解释最终训练。"),
    ]),
    "orca_refit_env.py": ("10 当前基础奖励覆盖层", [
        ("BASE_WEIGHTS =", "本轮非零基础项只有有效保持+15、终止-50；其余分量虽记录但权重为零。"),
        ("        reward=sum(self.weights", "覆盖父类奖励，而非叠加；上层Acquisition再加入前3秒辅助项。"),
    ]),
    "orca_acquisition_env.py": ("11 前3秒接近奖励", [
        ("        acquisition = jp.where", "仅前3秒奖励接近和配对接触；不计算与视频角度的距离。S04残差配方shaping=0关闭此项。"),
    ]),
    "orca_hic53_env.py": ("12 手物与参考契约", [
        ("def task_packet(model, reference):", "从参考读取物体姿态和共同起手，强制53mm、空间比例1、有效关节范围。"),
        ("        packet = task_packet(compiled_model, reference)", "即使pure分支不读逐帧参考，也共用这些任务元数据；不能说纯PPO场景完全与视频无关。"),
    ]),
    "orca_matched_env.py": ("13 观测与模拟触觉", [
        ("    def _obs(self, data, info):", "当前策略读关节/物体状态和模拟触觉，不读画面；去除actor中的保持计数信息。"),
        ("    def step(self, state, action):", "接触力加噪、滤波并延迟提供给策略；这些是仿真传感器，不是视频识别出的力。"),
    ]),
    "orca_native_physics.py": ("14 JAX策略与原生MuJoCo物理", [
        ("    def pack(self, data):", "保存完整积分状态含warm-start；uint32仅用于无损携带float64字节，不能直接当物理数值。"),
        ("    def call(self, mode, *arrays):", "通过pure_callback调用CPU原生MuJoCo，网络更新仍由JAX/Brax完成；不是全GPU物理。"),
    ]),
    "orca_suite_env.py": ("15 扰动条件与5ms接触审查", [
        ("BENCHMARKS =", "只改变所列物理条件；B0训练，其他条件直接测试，不逐条件重新训练。"),
        ("    def sample_contacts(self, data):", "检查每个物理子步的最大穿透、禁接触和指部峰值力，避免20Hz抽样漏掉短暂穿透。"),
        ("            if variant == \"linear\":", "线性消融保留起点/终点/合拢时间，仅删除视频中间动作，隔离轨迹形状贡献。"),
        ("        nxt = super().step(incoming, action * self.residual_gain", "零参考将gain设为0；半修正设为0.5。训练好的网络不必重训就能做执行端消融。"),
    ]),
    "run_orca_suite_job.py": ("16 S02训练与正式评分", [
        ("def summarize(trace, env, seeds):", "先算连续保持，再用撤托后的每5ms接触审查判定清洁成功；高reward不等于成功。"),
        ("    elif method in (\"guided\", \"bc\"):", "guided和仅BC共享4000步BC初始化；guided随后调用PPO，bc在初始化后停止。"),
        ("        train.train(environment=env", "调用Brax现成PPO；本项目负责环境、先验接口、初始化及评估，不自行另写PPO公式。"),
    ]),
    "run_orca_s03_job.py": ("17 S04所用权重的训练来源", [
        ("def validate_job(job):", "约束训练预算、训练场景和开发种子，避免训练回调读取最终测试初态。"),
        ("    eval_env = OrcaS03Grasp", "训练环境不采集昂贵子步审查；开发评估单独开启，物理任务与动作接口一致。"),
        ("        (directory / \"params.pkl\").write_bytes(blob)", "保留各固定步数权重；S04取163840步，而不是按新测试成功率挑最好权重。"),
        ("        train.train(environment=env", "PPO从交互轨迹学习；guided从BC参数起步，residual学修正，二者都实际执行PPO更新。"),
    ]),
    "run_orca_s03_eval.py": ("18 S04正式回放评估入口", [
        ("def validate_job(job):", "评估只加载权重不训练；zero无需权重，其他路线必须提供匹配检查点。"),
        ("    def save(name, environment, result):", "记录全部回合指标，仅保存前4个可视轨迹；视频数不等于统计样本数。"),
    ]),
    "run_orca_s04.py": ("19 新初态评估矩阵与统计", [
        ("def jobs(recipes):", "生成41个任务：4种学习方法各10个训练种子，加1个确定性零修正控制器。"),
        ("                leaf = \"bc_params.pkl\"", "BC取预训练权重；主三组取同为163840步检查点，非S03最终长训练权重。"),
        ("def load_protocol():", "重放前核验冻结的源码、权重与数据；这也是不直接改原训练文件注释的原因。"),
    ]),
    "report_orca_suite.py": ("20 统计方法", [
        ("def bootstrap_interval(values, seed=711, draws=5000):", "以训练种子和共同测试初态两个轴重新抽样，给出不确定区间；不是把所有回合当独立训练。"),
        ("def sign_flip_pvalue(difference):", "对10个配对种子差值枚举正负号，检查平均提升是否容易由随机波动解释。"),
        ("def holm(values):", "同时检验多项主张时校正p值；S02用5项、S04用2项，不混为同一检验家族。"),
    ]),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_tokens(text):
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}
    return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(text).readline)
            if t.type not in ignored]


def annotate(text, filename, title, notes):
    lines = text.splitlines(keepends=True)
    inserts = {}
    for anchor, explanation in notes:
        matches = [i for i, line in enumerate(lines) if line.startswith(anchor)]
        if len(matches) != 1:
            raise ValueError(f"Anchor must be unique: {filename}: {anchor}")
        index = matches[0]
        indent = lines[index][:len(lines[index])-len(lines[index].lstrip())]
        inserts[index] = indent + "# 导读：" + explanation + "\n"
    header = (f"# 中文阅读副本：{title}\n# 原始入口：scripts/{filename}\n"
              "# 仅用于阅读，保留原路径假设；不要从此副本启动训练。运行请使用原始入口。\n"
              "# 本副本只增加注释，生成时核验AST与非注释词法序列完全一致。\n")
    # Keep a source shebang as its first line.
    start = 1 if lines and lines[0].startswith("#!") else 0
    result = "".join(lines[:start]) + header
    result += "".join(inserts.get(i, "") + line for i, line in enumerate(lines) if i >= start)
    if ast.dump(ast.parse(text)) != ast.dump(ast.parse(result)) or code_tokens(text) != code_tokens(result):
        raise ValueError("Annotations changed executable content: " + filename)
    return result


def main():
    for name in ("core_source", "data", "figures"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    inputs, files, sources = {}, {}, []
    protocol_path = BASE / "S04/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    inputs[str(protocol_path.relative_to(ROOT))] = sha(protocol_path)
    for filename, (title, notes) in NOTES.items():
        path = ROOT / "scripts" / filename
        original_hash = sha(path)
        inputs[str(path.relative_to(ROOT))] = original_hash
        frozen = protocol["source_hashes"].get(str(path.relative_to(ROOT)))
        if frozen is not None and frozen != original_hash:
            raise ValueError("Current source differs from S04 frozen source: " + filename)
        output = OUT / "core_source" / filename
        output.write_text(annotate(path.read_text(), filename, title, notes))
        files[str(output.relative_to(OUT))] = sha(output)
        sources.append(dict(title=title, original=str(path.relative_to(ROOT)),
                            snapshot=str(output.relative_to(OUT)), source_sha256=original_hash,
                            snapshot_sha256=sha(output), comments=len(notes),
                            ast_and_code_tokens_equal=True, matches_S04_frozen=frozen is not None))
    copied = {}
    for path in sorted((BASE / "S02_S04_comparison/figures").glob("*")):
        if path.suffix in (".png", ".pdf"):
            copied["figures/" + path.name] = path
    for path in sorted((BASE / "S02_S04_comparison/tables").glob("*.csv")):
        copied["data/" + path.name] = path
    copied.update({
        "data/S03_historical_conditions.csv": BASE / "S04/report/historical_conditions.csv",
        "data/analysis_manifest.json": BASE / "S02_S04_comparison/manifest.json",
        "data/D01_dataset_manifest.json": ROOT / "artifacts/orca_cube_acquisition/D01_quality_demos/dataset_manifest.json",
        "data/E01_refinement_report.json": ROOT / "artifacts/orca_cube_acquisition/E01_flex50_refinement/report.json",
        "data/S03_selected_recipes.json": BASE / "S03/selected_recipes.json",
    })
    for dest, path in copied.items():
        inputs[str(path.relative_to(ROOT))] = sha(path)
        shutil.copyfile(path, OUT / dest)
        files[dest] = sha(OUT / dest)
        if files[dest] != inputs[str(path.relative_to(ROOT))]:
            raise ValueError("Copy mismatch: " + dest)
    # A single human-readable table appendix, without pooling different conditions.
    with (OUT / "data/aggregate.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    names = {"ppo": "纯PPO", "guided": "BC后PPO", "residual": "参考+残差PPO", "bc": "仅BC",
             "zero": "校正参考零修正", "linear": "线性参考+残差", "unrefined": "未校正参考+残差",
             "zero_linear": "线性参考零修正", "zero_unrefined": "未校正参考零修正", "half_residual": "修正减半"}
    table = ["# 完整数据表", "", "[返回总报告](report.md)。保留实验来源，不把不同场景合成一个成功率。", "",
             "清洁成功率同时满足30秒任务要求与撤托后的5ms接触审查；区间为95%交叉bootstrap区间。", ""]
    for run in ("S02", "S04"):
        table += [f"## {run}", "", "| 条件 | 方法 | 策略数/回合数 | 原始成功率 | 清洁成功率 | 95%区间 | 平均最长保持(s) | 平均回合峰值力(N) |",
                  "|---|---|---:|---:|---:|---|---:|---:|"]
        for r in rows:
            if r["round"] == run:
                table.append(f"| {r['condition']} | {names[r['method']]} | {r['policies']}/{r['episodes']} | "
                             f"{float(r['raw']):.2%} | {float(r['clean']):.2%} | "
                             f"[{float(r['ci_low']):.2%}, {float(r['ci_high']):.2%}] | "
                             f"{float(r['hold_mean_sec']):.2f} | {float(r['force_mean_n']):.2f} |")
        table.append("")
    (OUT / "all_results.md").write_text("\n".join(table)+"\n")
    files["all_results.md"] = sha(OUT / "all_results.md")
    index = ["# 核心源码中文注释版", "", "[阅读源码讲解](../source_guide.md) / [阅读完整实验报告](../report.md)", "",
             "这是完整文件的只读教学快照，不是新的训练入口。原文件未修改；快照仅增加中文注释，AST和非注释词法序列均核验一致。", "",
             "S04冻结列为否，只表示不在S04协议的冻结源码清单内，例如更早的数据准备脚本；不代表已核实为历史运行时的精确版本。", "",
             "| 顺序与作用 | 注释版 | 原始源码 | 本次注释处数 | 与S04冻结版本一致 |", "|---|---|---|---:|---|"]
    for r in sources:
        name = Path(r["original"]).name
        index.append(f"| {r['title']} | [{name}]({name}) | [原文件](../../../../{r['original']}) | {r['comments']} | {'是' if r['matches_S04_frozen'] else '不在清单'} |")
    (OUT / "core_source/README.md").write_text("\n".join(index)+"\n")
    files["core_source/README.md"] = sha(OUT / "core_source/README.md")
    for name in ("README.md", "report.md", "source_guide.md", "data_guide.md"):
        if (OUT / name).exists():
            files[name] = sha(OUT / name)
    for path, expected in inputs.items():
        if sha(ROOT / path) != expected:
            raise ValueError("Source changed during packaging: " + path)
    manifest = dict(purpose="Integrated experiment narrative and annotated reading package",
                    generated_date="2026-10-09", source_files=sources, input_sha256=inputs,
                    output_sha256=files, no_training=True, originals_unchanged=True,
                    package_script_sha256=sha(Path(__file__)))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(sources=len(sources), comment_blocks=sum(r["comments"] for r in sources),
                         outputs=len(files), verified=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
