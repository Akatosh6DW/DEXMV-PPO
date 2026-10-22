"""Select a can grasp segment using source geometry only, before Orca trials."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from prepare_cylinder_s02_source import digest

ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT/"DexMV/hand_imitation/env/models/assets/ycb/collision/005_tomato_soup_can/005_tomato_soup_can_1.stl"


def candidates(directory):
    mesh = trimesh.load(MESH, force="mesh")
    center, extents = mesh.bounds.mean(0), mesh.extents
    radius, half_height = extents[0]/2, extents[2]/2
    records = []
    manifest = json.loads((directory/"manifest.json").read_text())
    for item in manifest["records"]:
        path = directory/item["packet"]
        if digest(path) != item["sha256"]:
            raise ValueError("Source changed")
        with np.load(path, allow_pickle=False) as data:
            points = data["fingertips_object_local"][:, :2]-center
        radial = np.linalg.norm(points[:, :, :2], axis=-1)
        normals = points[:, :, :2]/np.maximum(radial[:, :, None], 1e-9)
        opposition = np.sum(normals[:, 0]*normals[:, 1], axis=-1)
        gap = np.abs(radial-radius)
        valid = (opposition < -.5) & (gap.max(1) < .015) & (np.abs(points[:, :, 2]).max(1) < half_height-.004)
        # A 0.3 s contiguous source segment rules out isolated pose coincidences.
        for end in range(89, len(points)):
            hold = slice(end-29, end+1)
            if not valid[hold].all():
                continue
            speed = np.linalg.norm(np.diff(points[hold], axis=0), axis=-1)
            if speed.max() > .008:
                continue
            score = float(gap[hold].mean()+speed.mean())
            records.append(dict(sequence=item["sequence"], packet=item["packet"], start=end-89, end=end,
                                geometry_score_m=score, mean_radial_gap_m=float(gap[hold].mean()),
                                max_frame_displacement_m=float(speed.max()),
                                mean_opposition_cosine=float(opposition[hold].mean())))
    return records, center, extents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int,
                        help="Use the best geometric window in this sequence; retain the full ranking")
    args = parser.parse_args()
    rows, center, extents = candidates(args.source)
    if not rows:
        raise ValueError("No source segment meets the declared geometric gates")
    rows.sort(key=lambda row: (row["geometry_score_m"], row["sequence"], row["end"]))
    eligible = rows if args.sequence is None else [row for row in rows if row['sequence'] == args.sequence]
    if not eligible:
        raise ValueError('Requested sequence has no geometrically eligible segment')
    selected = eligible[0]
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source/selected["packet"]
    with np.load(source, allow_pickle=False) as data:
        selection = slice(selected["start"], selected["end"]+1)
        points = data["fingertips_object_local"][selection]-center
        time = data["time"][selection]-data["time"][selected["start"]]
    np.savez_compressed(args.output/"selected_segment.npz", fingertips_centered=points, time=time,
                        source_center=center, source_extents_m=extents,
                        source_sequence=selected["sequence"], source_indices=np.arange(selected["start"], selected["end"]+1))
    report = dict(selected=selected, candidates=rows, source_sha256=digest(source),
                  manifest_sha256=digest(args.source/"manifest.json"), mesh_sha256=digest(MESH),
                  source_center_m=center.tolist(), source_extents_m=extents.tolist(),
                  target_extents_m=[.053]*3,
                  criterion="Opposing thumb/index on cylindrical side, near surface, smooth for 0.3 s; geometry-only minimum gap",
                  robot_rewards_used=False, robot_rollouts_used=False,
                  requested_sequence=args.sequence,
                  note="Adroit marker geometry from published processed data, not independently verified human pose truth")
    (args.output/"selection.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(selected=selected, candidates=len(rows))), flush=True)


if __name__ == "__main__":
    main()
