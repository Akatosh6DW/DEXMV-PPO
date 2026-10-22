"""Resolve actual evaluated policy checkpoints, not arbitrary latest directories."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "artifacts/orca_benchmark_suite"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def eval_roots():
    roots = {"cube_S02": SUITE / "S02/eval", "cube_S04": SUITE / "S04/eval",
             "cylinder_S02": ROOT / "artifacts/orca_cylinder_s02/eval",
             "S05": SUITE / "S05/E01_independent", "S06": SUITE / "S06/E01_independent"}
    roots.update({f"S{i:02}": SUITE / f"S{i:02}/eval" for i in range(7, 11)})
    roots["S09_reward_iteration"] = SUITE / "S09/reward_iteration/eval"
    for path in (SUITE / "S09/bc_repair").rglob("eval"):
        if path.parent.name.startswith("confirmation"):
            roots["S09_"+path.parent.name] = path
    return roots


def local_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def inputs(config, label):
    found = {}
    for key in ("model", "reference", "raw_reference", "selection"):
        if config.get(key):
            path = local_path(config[key])
            if path.is_file():
                found[key] = path
    if config.get("input"):
        folder = local_path(config["input"])
        for key, name in (("model", "model.xml"), ("reference", "reference.npz")):
            if key not in found and (folder / name).is_file():
                found[key] = folder / name
    if label == "S10":
        found.setdefault("model", SUITE / "S10/inputs/model.xml")
        found.setdefault("reference", SUITE / "S09/inputs/reference.npz")
    if label.startswith("S09"):
        found.setdefault("model", SUITE / "S09/inputs/model.xml")
        found.setdefault("reference", SUITE / "S09/inputs/reference.npz")
    if config.get("raw_reference_sha256") and "raw_reference" not in found:
        candidates = [found["model"].parent / "reference.npz"] if "model" in found else []
        if label == "cylinder_S02":
            candidates.extend((ROOT / "artifacts/orca_cylinder_s02/inputs").glob("*.npz"))
        for candidate in candidates:
            if candidate.is_file() and digest(candidate) == config["raw_reference_sha256"]:
                found["raw_reference"] = candidate
                break
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(kind="frozen evaluation input inventory", policies={}, inputs={}, evaluations=[], errors=[])
    for label, root in eval_roots().items():
        configs = sorted(root.rglob("run_config.json"))
        if not configs:
            report["errors"].append(dict(group=label, error="No evaluation configurations"))
        for path in configs:
            config = json.loads(path.read_text())
            record = dict(group=label, config=str(path.relative_to(ROOT)),
                          config_sha256=digest(path), method=config.get("method"), inputs={})
            checkpoint = config.get("checkpoint")
            exact_zero = (label == "S10" and path.parent.name in
                          ("reference_exact_s0", "reference_exact_quality_s0"))
            if exact_zero:
                checkpoint = str(path.parent / "params.pkl")
                record["policy_role"] = "untrained deterministic zero-residual diagnostic"
            if checkpoint:
                checkpoint = local_path(checkpoint)
                if not checkpoint.is_file():
                    report["errors"].append(dict(config=record["config"], error="Missing checkpoint", path=str(checkpoint)))
                    continue
                key = digest(checkpoint)
                expected = config.get("checkpoint_sha256")
                if expected and key != expected:
                    report["errors"].append(dict(config=record["config"], error="Checkpoint hash mismatch"))
                    continue
                entry = report["policies"].setdefault(key, dict(bytes=checkpoint.stat().st_size, paths=[]))
                relative = str(checkpoint.relative_to(ROOT))
                if relative not in entry["paths"]:
                    entry["paths"].append(relative)
                record["policy_sha256"] = key
            else:
                record["policy_sha256"] = None
                if config.get("method") not in ("reference", "collect", "zero"):
                    report["errors"].append(dict(config=record["config"], error="Non-reference evaluation lacks checkpoint"))
            for role, source in inputs(config, label).items():
                if not source.is_file():
                    report["errors"].append(dict(config=record["config"], error="Missing input", path=str(source)))
                    continue
                key = digest(source)
                expected = config.get(role+"_sha256")
                if expected and key != expected:
                    report["errors"].append(dict(config=record["config"], error="Input hash mismatch", role=role))
                    continue
                entry = report["inputs"].setdefault(key, dict(bytes=source.stat().st_size, paths=[]))
                relative = str(source.relative_to(ROOT))
                if relative not in entry["paths"]:
                    entry["paths"].append(relative)
                record["inputs"][role] = key
            for required in ("model", "reference"):
                if required not in record["inputs"]:
                    report["errors"].append(dict(config=record["config"], error="Unresolved input", role=required))
            if config.get("raw_reference_sha256") and "raw_reference" not in record["inputs"]:
                report["errors"].append(dict(config=record["config"], error="Unresolved raw reference"))
            report["evaluations"].append(record)
    report["summary"] = dict(evaluations=len(report["evaluations"]),
        unique_policies=len(report["policies"]), policy_bytes=sum(v["bytes"] for v in report["policies"].values()),
        unique_inputs=len(report["inputs"]), input_bytes=sum(v["bytes"] for v in report["inputs"].values()),
        errors=len(report["errors"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["summary"], indent=2))
    if report["errors"]:
        print(json.dumps(report["errors"][:10], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
