"""Final S04 artifact audit; does not change the frozen experiment or scores."""

import csv
import itertools
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image

from run_orca_s04 import OUT, ROOT, collect, digest, load_protocol


def main():
    protocol = load_protocol()
    groups, per_seed, episodes = collect(protocol)
    stats = json.loads((OUT / "report/statistics.json").read_text())
    assert stats["complete"] and stats["episodes"] == len(episodes) == 4100
    assert stats["protocol_sha256"] == digest(OUT / "protocol.json")
    for record in stats["aggregate"]:
        rows = [r for _, data in groups[record["method"]] for r in data]
        assert record["episodes"] == len(rows)
        for key, source in (("clean", "clean_success30"), ("raw", "success30"),
                            ("full_clean", "full_episode_clean_success30")):
            np.testing.assert_allclose(record[key], np.mean([r[source] for r in rows]))
    base = np.array([[r["clean_success30"] for r in rows] for _, rows in groups["ppo"]], float)
    raw_p = []
    for record, method in zip(stats["comparisons"], ("guided", "residual")):
        matrix = np.array([[r["clean_success30"] for r in rows] for _, rows in groups[method]], float)
        delta = matrix-base
        values = delta.mean(axis=1)
        count = sum(abs(sum(s*v for s, v in zip(signs, values))/10) >= abs(values.mean())-1e-12
                    for signs in itertools.product((-1, 1), repeat=10))
        p = count/1024
        np.testing.assert_allclose(record["difference_pp"], delta.mean()*100)
        np.testing.assert_allclose(record["p"], p)
        raw_p.append(p)
    order = np.argsort(raw_p)
    adjusted = np.zeros(2)
    adjusted[order[0]] = min(1., 2*raw_p[order[0]])
    adjusted[order[1]] = max(adjusted[order[0]], raw_p[order[1]])
    np.testing.assert_allclose(adjusted, [r["holm_2_p"] for r in stats["comparisons"]])
    for name, expected in (("episodes", 4100), ("per_seed", 41), ("aggregate", 5),
                           ("paired", 2), ("historical_conditions", 30), ("learning_curves", 90)):
        with (OUT / "report" / (name+".csv")).open() as stream:
            assert len(list(csv.DictReader(stream))) == expected
    for name in ("comparison", "learning_curves"):
        pixels = np.asarray(Image.open(OUT / "report" / (name+".png")))
        assert min(pixels.shape[:2]) >= 600 and pixels.std() > 5
    manifest = json.loads((OUT / "videos/manifest.json").read_text())
    assert manifest["complete"] and len(manifest["cases"]) == 5
    assert manifest["protocol_sha256"] == digest(OUT / "protocol.json")
    assert {c["method"] for c in manifest["cases"]} == set(groups)
    videos = 0
    for case in manifest["cases"]:
        assert case["seed"] == 21001 and case["test_seed"] == 81000
        expected = groups[case["method"]][0][1][0]
        assert case["outcome"] == expected
        d = OUT / "videos" / case["method"]
        render = json.loads((d / "report.json").read_text())
        trace = OUT / "eval" / case["method"] / "s21001/evaluation/preview_trajectories.npz"
        assert render["physics_arrays_match"] and render["no_physics_resimulation"]
        assert render["same_saved_states_both_modes"] and render["trace_sha256"] == digest(trace)
        assert render["camera"]["azimuth"] == 225
        for mode, video in case["videos"].items():
            path = OUT / video["path"]
            assert digest(path) == video["sha256"]
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration",
                "-of", "json", str(path)], text=True))["streams"][0]
            assert probe == video["probe"]
            assert int(probe["nb_read_frames"]) == 800 and probe["r_frame_rate"] == "20/1"
            assert (probe["width"], probe["height"]) == (960, 720)
            assert abs(float(probe["duration"])-40) < .01
            assert render["pixel_checks"][mode]["mean_frame_change"] > .01
            videos += 1
    assert videos == 10
    docs = ROOT / "docs/experiments/S04"
    for path in docs.glob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if not link.startswith(("http:", "https:", "#")):
                assert (path.parent / link.split("#")[0]).exists(), (path, link)
    result = dict(complete=True, evaluation_jobs=41, episodes=4100, independently_rescored_preview_cases=164,
                  paired_states=True, exact_p_values_recomputed=True, videos=videos,
                  source_inputs_verified=len(protocol["source_hashes"]), document_links_valid=True,
                  manual_visual_review="Recorded separately in docs/experiments/S04/execution.md",
                  protocol_sha256=digest(OUT / "protocol.json"), statistics_sha256=digest(OUT / "report/statistics.json"),
                  video_manifest_sha256=digest(OUT / "videos/manifest.json"), audit_source_sha256=digest(Path(__file__)))
    (OUT / "delivery_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
