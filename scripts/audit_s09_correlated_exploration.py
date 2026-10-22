"""Bind correlation metadata to BC source, then audit full-target deployment."""

import json
from pathlib import Path
import pickle
import sys

import numpy as np

import audit_s09_bc_repair as base
from retarget_s09_reference import digest
from run_s09_correlated_exploration import head_correlation


def main():
    folder = Path(sys.argv[sys.argv.index("--run")+1])
    config = json.loads((folder/"run_config.json").read_text())
    meta = json.loads((folder/"correlated_exploration_config.json").read_text())
    source = Path(meta["source_checkpoint"])
    assert digest(source) == meta["source_checkpoint_sha256"] == config["checkpoint_sha256"]
    assert meta["source_sha256"] == config["sources"]["run_s09_correlated_exploration.py"]
    for name, sha in config["sources"].items():
        assert digest(folder/"sources"/name) == sha
    old = pickle.loads(source.read_bytes())
    matrix = head_correlation(old[1]["params"]["hidden_2"]["kernel"], meta["correlation_strength"])
    np.testing.assert_array_equal(matrix, meta["correlation_matrix"])
    constraint = json.loads((folder/"constraint_config.json").read_text())
    assert constraint["teacher_kl_weight"] == 0
    assert meta["mean_network_and_critic_unchanged"] and meta["independently_resampled_every_step"]
    assert config["method"] == "guided" and config["orientation"] == "uniform"
    base.main()


if __name__ == "__main__":
    main()
