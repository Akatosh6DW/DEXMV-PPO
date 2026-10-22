"""Snapshot public source cards and verify local archives against HF LFS hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

from prepare_oakink_cap_source import SEQUENCE


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(16 * 1024**2 + 1)
    if len(data) > 16 * 1024**2:
        raise ValueError("Metadata response unexpectedly large")
    return data


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("artifacts/orca_formal/cap_source"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repos = {"official": "kelvin34501/OakInk-v2", "derived": "MIT-Media-Lab/oakink2-annotations-v1"}
    report = {}
    for label, repo in repos.items():
        info = json.loads(fetch(f"https://huggingface.co/api/datasets/{repo}?blobs=true"))
        revision = info["sha"]
        card_url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md"
        card = fetch(card_url)
        (args.output / f"{label}_README.md").write_bytes(card)
        (args.output / f"{label}_metadata.json").write_text(json.dumps(info, indent=2) + "\n")
        verified = []
        expected = ({"anno_preview.pkl": f"anno_preview/{SEQUENCE}.pkl", "video_frames.tar": f"data/{SEQUENCE}.tar",
                     "program.tar": "program.tar", "object_raw.tar": "object_raw.tar"} if label == "official" else {})
        if label == "derived":
            expected_hashes = {file_digest(args.source / name): name for name in
                               ("derived_annotations.tar", "derived_episode_index.npz")}
            for item in info["siblings"]:
                sha = item.get("lfs", {}).get("sha256")
                if sha in expected_hashes:
                    expected[expected_hashes[sha]] = item["rfilename"]
            if len(expected) != 2:
                raise ValueError("Cannot bind both local derived archives to published LFS objects")
        for local, remote in expected.items():
            item = next(i for i in info["siblings"] if i["rfilename"] == remote)
            actual = file_digest(args.source / local)
            if actual != item.get("lfs", {}).get("sha256"):
                raise ValueError("Local source differs from published LFS object: " + local)
            verified.append(dict(local=local, remote=remote, sha256=actual,
                                 url=f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(remote)}"))
        report[label] = dict(repository=repo, revision=revision, license=info.get("cardData", {}).get("license"),
                             readme_url=card_url, readme_sha256=hashlib.sha256(card).hexdigest(), verified=verified)
        print(label, revision, len(verified), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
