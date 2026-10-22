"""Use the exact source visual mesh with a measured, source-sized box proxy."""

import argparse
import copy
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial import HalfspaceIntersection

from fit_orca_collision import distance_to_mesh, mesh_scene
from prepare_hic_orca_object import digest, SOURCE, compiled_vertices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("box", "clean-convex"), default="box")
    p.add_argument("--timestep", type=float)
    args = p.parse_args()
    if args.timestep is not None and (not np.isfinite(args.timestep) or not .001 <= args.timestep <= .01):
        p.error("timestep must lie in [.001, .01]")
    off = SOURCE / "models_motion/20/Model_Object_Cube.OFF"
    mesh = trimesh.load(off, force="mesh")
    mesh.apply_scale(.001)
    half = mesh.extents * .5
    assert np.allclose(mesh.bounds.mean(0), 0, atol=1e-9)
    corners = np.array(list(itertools.product(*zip(-half, half))))
    corner_errors = distance_to_mesh(mesh_scene(mesh), corners)
    points, _ = trimesh.sample.sample_surface(mesh, 10000, seed=101)
    distances = np.abs(np.max(np.abs(points) - half, axis=1))
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(source=str(off), source_sha256=digest(off),
                  parent_model_sha256=digest(args.model), extents_m=mesh.extents.tolist(), mode=args.mode,
                  source_surface_to_box_p95_mm=float(np.quantile(distances, .95) * 1000),
                  source_surface_to_box_max_sampled_mm=float(distances.max() * 1000),
                  box_corner_to_source_max_mm=float(corner_errors.max() * 1000),
                  box_volume_excess_fraction=float(np.prod(mesh.extents) / mesh.volume - 1))
    proxy = None
    if args.mode == "clean-convex":
        normals = np.array([v for v in itertools.product([-1, 0, 1], repeat=3) if any(v)], dtype=float)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        offsets = np.max(mesh.vertices @ normals.T, axis=0)
        vertices = HalfspaceIntersection(np.c_[normals, -offsets], np.zeros(3)).intersections
        proxy = trimesh.points.PointCloud(vertices).convex_hull
        sample, _ = trimesh.sample.sample_surface(proxy, 10000, seed=102)
        forward = distance_to_mesh(mesh_scene(proxy), points)
        reverse = distance_to_mesh(mesh_scene(mesh), np.r_[sample, proxy.vertices])
        result.update(source_to_proxy_p95_mm=float(np.quantile(forward, .95) * 1000),
                      proxy_to_source_p95_mm=float(np.quantile(reverse, .95) * 1000),
                      bidirectional_max_sampled_mm=float(max(forward.max(), reverse.max()) * 1000),
                      proxy_vertices=len(proxy.vertices), proxy_faces=len(proxy.faces))
        accepted = result["bidirectional_max_sampled_mm"] <= 2. and len(proxy.vertices) <= 64
    else:
        accepted = corner_errors.max() <= .002
    result["accepted"] = bool(accepted)
    if not accepted:
        (args.output / "object_proxy_report.json").write_text(json.dumps(result, indent=2) + "\n")
        raise ValueError("Proxy rejected: " + json.dumps(result))
    root = ET.parse(args.model)
    if args.timestep is not None:
        root.find("option").set("timestep", str(args.timestep))
    for asset in root.findall("asset/mesh"):
        file = Path(asset.get("file"))
        if not file.is_absolute():
            asset.set("file", str((args.model.parent / file).resolve()))
    cube = root.find(".//geom[@name='cube']")
    visual = copy.deepcopy(cube)
    visual.set("name", "hic20_cube_visual")
    visual.set("contype", "0")
    visual.set("conaffinity", "0")
    visual.set("mass", "0")
    visual.set("group", "2")
    root.find(".//body[@name='cube']").append(visual)
    if proxy is None:
        cube.attrib.pop("mesh")
        cube.set("type", "box")
        cube.set("size", " ".join(map(str, half)))
    else:
        proxy.export(args.output / "hic20_clean_convex.stl")
        ET.SubElement(root.find("asset"), "mesh", name="hic20_clean_convex",
                      file="hic20_clean_convex.stl", maxhullvert="64")
        cube.set("mesh", "hic20_clean_convex")
    cube.set("group", "3")
    root.write(args.output / "cube.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(args.output / "cube.xml"))
    compiled_size = 2 * model.geom("cube").size if proxy is None else np.ptp(compiled_vertices(model), axis=0)
    np.testing.assert_allclose(compiled_size, mesh.extents, atol=1e-7)
    np.testing.assert_allclose(np.ptp(compiled_vertices(model, "hic20_cube_visual"), axis=0), mesh.extents, atol=1e-7)
    result.update(model_sha256=digest(args.output / "cube.xml"),
                  simulation_timestep_sec=float(model.opt.timestep),
                  note="Same object and metric dimensions; measured collision approximation. Source mesh remains the visual shape.")
    (args.output / "object_proxy_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
