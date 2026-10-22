"""Source-mesh convex decomposition with exposed-surface and cavity checks."""

import argparse
import json
from pathlib import Path

import coacd
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
import trimesh

from prepare_hic_orca_object import digest


def scene(mesh):
    result = o3d.t.geometry.RaycastingScene(nthreads=1)
    result.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
                         o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32))
    return result


def distance(surface, points):
    return surface.compute_distance(o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)).numpy()


def inside(points, part, tolerance=0.):
    planes = ConvexHull(part.vertices).equations
    return np.max(points @ planes[:, :3].T + planes[:, 3], axis=1) <= tolerance


def metrics(source, parts):
    source_points, _ = trimesh.sample.sample_surface(source, 10000, seed=1804)
    forward = np.min([distance(scene(p), source_points) for p in parts], axis=0)
    reverse, lateral = [], []
    original = scene(source)
    for i, part in enumerate(parts):
        points, faces = trimesh.sample.sample_surface(part, 1000, seed=1804 + i)
        normals = part.face_normals[faces]
        outward = points + normals * 1e-6
        exposed = np.ones(len(points), bool)
        for j, other in enumerate(parts):
            if i != j:
                exposed &= ~inside(outward, other)
        if exposed.any():
            distances = distance(original, points[exposed])
            reverse.extend(distances.tolist())
            lateral.extend(distances[abs(normals[exposed, 2]) < .9].tolist())
    return dict(source_to_proxy_p95_mm=float(1000 * np.percentile(forward, 95)),
                source_to_proxy_max_mm=float(1000 * np.max(forward)),
                exposed_proxy_to_source_p95_mm=float(1000 * np.percentile(reverse, 95)),
                exposed_proxy_to_source_max_mm=float(1000 * np.max(reverse)),
                exposed_lateral_proxy_to_source_p95_mm=float(1000 * np.percentile(lateral, 95)) if lateral else None,
                exposed_lateral_proxy_to_source_max_mm=float(1000 * np.max(lateral)) if lateral else None,
                sampling="10000 source samples to nearest part surface; 1000 per proxy, outward-step union test removes overlapping/abutting internal faces; sampled diagnostics, not an exact Hausdorff bound")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/orca_formal/CAP01_source_inspection"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objects", choices=("cap", "bottle"), nargs="+", default=["cap", "bottle"])
    parser.add_argument("--preprocess-resolution", type=int, default=60)
    args = parser.parse_args()
    if args.preprocess_resolution < 20 or len(set(args.objects)) != len(args.objects):
        parser.error("Expected distinct objects and preprocess resolution >=20")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "preparation_source.py").write_bytes(Path(__file__).read_bytes())
    reports = {}
    coacd.set_log_level("warn")
    options = dict(threshold=.025, max_convex_hull=32, resolution=2000, mcts_iterations=50,
                   mcts_nodes=15, seed=1804, preprocess_resolution=args.preprocess_resolution, decimate=True, max_ch_vertex=48)
    for name in args.objects:
        mesh = trimesh.load(args.source / f"{name}.ply", force="mesh", process=True)
        mesh.export(args.output / f"{name}_source.stl")
        raw = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), **options)
        parts = [trimesh.Trimesh(vertices=v, faces=f, process=True) for v, f in raw]
        directory = args.output / name
        directory.mkdir()
        for i, part in enumerate(parts):
            part.export(directory / f"part_{i:02d}.stl")
        report = dict(parts=len(parts), source_sha256=digest(args.source / f"{name}.ply"),
                      source_watertight=bool(mesh.is_watertight), source_bounds_m=mesh.bounds.tolist(),
                      source_volume_m3=float(mesh.volume), hull_volume_m3=float(mesh.convex_hull.volume),
                      hull_metrics=metrics(mesh, [mesh.convex_hull]), decomposition_metrics=metrics(mesh, parts),
                      asset_sha256={p.name: digest(p) for p in sorted(directory.glob("*.stl"))})
        if name == "cap":
            heights = np.linspace(.043, .055, 7)
            probes = np.c_[np.zeros(len(heights)), np.zeros(len(heights)), heights]
            report["cavity_probes_m"] = probes.tolist()
            report["cavity_filled_by_proxy"] = np.any([inside(probes, part) for part in parts], axis=0).tolist()
        reports[name] = report
        print(json.dumps(dict(object=name, **report)), flush=True)
        (args.output / "report.json").write_text(json.dumps(dict(options=options, objects=reports,
                                                                 requested_objects=args.objects,
                                                                 complete=set(reports) == set(args.objects)), indent=2) + "\n")


if __name__ == "__main__":
    main()
