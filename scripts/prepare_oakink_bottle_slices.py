"""Piecewise convex outer bottle profile without voxel preprocessing."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from prepare_hic_orca_object import digest
from prepare_oakink_cap_collision import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/orca_formal/CAP01_source_inspection/bottle.ply"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "bottle").mkdir()
    (args.output / "preparation_source.py").write_bytes(Path(__file__).read_bytes())
    mesh = trimesh.load(args.source, force="mesh", process=True)
    boundaries = np.r_[mesh.bounds[0, 2] - 1e-7, -.04, -.02, 0, .02, .03,
                       np.arange(.032, .0581, .002), mesh.bounds[1, 2] + 1e-7]
    parts = []
    for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        vertices, faces, _ = trimesh.intersections.slice_faces_plane(mesh.vertices, mesh.faces,
                                                                    np.array([0, 0, 1.]), np.array([0, 0, lo]))
        vertices, faces, _ = trimesh.intersections.slice_faces_plane(vertices, faces,
                                                                    np.array([0, 0, -1.]), np.array([0, 0, hi]))
        part = trimesh.Trimesh(vertices, faces, process=True).convex_hull
        if part.volume <= 0 or not part.is_watertight:
            raise ValueError("Invalid bottle slice")
        part.export(args.output / "bottle" / f"part_{i:02d}.stl")
        parts.append(part)
    mesh.export(args.output / "bottle_source.stl")
    report = dict(complete=True, objects=dict(bottle=dict(parts=len(parts), source_sha256=digest(args.source),
                                                         boundaries_m=boundaries.tolist(),
                                                         decomposition_metrics=metrics(mesh, parts))),
                  method="Convex hull per exact mesh-clipped z slab; source coordinates and extents unchanged",
                  limitation="Bottle interior filled, external shoulder/neck profile preserved; no measured thread tooth geometry")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
