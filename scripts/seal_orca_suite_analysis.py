"""Record analysis rules before any final-test job has started."""

import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "artifacts/orca_benchmark_suite/S02"


def main():
    if (SUITE / "eval").exists():
        raise RuntimeError("Final evaluations have started; do not claim prospective registration")
    paths = [SUITE / "statistics_plan.json", SUITE / "protocol.json"]
    paths += [ROOT / "scripts" / name for name in (
        "report_orca_suite.py", "render_orca_suite.py", "audit_orca_suite.py", "plot_orca_suite.py",
        "check_orca_suite_resets.py", "test_orca_suite_report.py", "test_orca_suite_audit.py")]
    paths += [ROOT / "docs" / name for name in (
        "orca_benchmark_suite_protocol.md", "orca_benchmark_data_guide.md")]
    record = dict(epoch_time=time.time(), final_evaluation_started=False,
                  training_completed=sorted(p.parent.name for p in (SUITE / "train").glob("*/summary.json")),
                  hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  note="Local prospective record, not an externally registered trial. Later tool bug fixes must be disclosed; no test-driven method changes.")
    with (SUITE / "analysis_seal.json").open("x") as stream:
        stream.write(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
