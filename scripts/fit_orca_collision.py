"""STL-derived primitive proxies, with independent surface samples and frozen inertia."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import trimesh

from build_orca_mjx_grasp_model import ORCA_V2, MJCF_DIR

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/orca_residual/models/cube.xml"


def rotation(quat):
    q = np.asarray(quat, float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def quat(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return q[[3, 0, 1, 2]]


def text(values):
    return " ".join(f"{v:.10g}" for v in np.atleast_1d(values))


def load_parts():
    assets = {e.get("name"): ORCA_V2 / e.get("file")
              for e in ET.parse(MJCF_DIR / "orcahand_right.mjcf").findall("asset/mesh")}
    parts = {}
    for body in ET.parse(MJCF_DIR / "orcahand_right_body.xml").iter("body"):
        meshes = []
        for geom in body.findall("geom"):
            path = assets[geom.get("mesh")]
            mesh = trimesh.load(path, force="mesh", process=True)
            vertices = np.asarray(mesh.vertices) * .001
            transform = rotation(np.fromstring(geom.get("quat", "1 0 0 0"), sep=" "))
            vertices = vertices @ transform.T + np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            meshes.append((path, trimesh.Trimesh(vertices, mesh.faces, process=False)))
        if meshes:
            parts[body.get("name")] = meshes
    return parts


def mesh_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
                        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32))
    return scene


def distance_to_mesh(scene, points):
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points), dtype=o3d.core.Dtype.Float32)).numpy()


def sdf(points, primitive):
    local = (points - primitive["pos"]) @ primitive["rot"]
    if primitive["type"] == "mesh":
        if "scene" not in primitive:
            primitive["scene"] = mesh_scene(primitive["mesh"])
            primitive["planes"] = ConvexHull(primitive["mesh"].vertices).equations
        planes = primitive["planes"]
        inside = np.max(local @ planes[:,:3].T+planes[:,3], axis=1) <= 1e-9
        return distance_to_mesh(primitive["scene"], local)*np.where(inside, -1, 1)
    if primitive["type"] == "box":
        q = np.abs(local) - primitive["size"]
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    if primitive["type"] == "sphere":
        return np.linalg.norm(local, axis=1) - primitive["size"][0]
    if primitive["type"] == "capsule":
        local = local.copy()
        local[:, 2] -= np.clip(local[:, 2], -primitive["size"][1], primitive["size"][1])
        return np.linalg.norm(local, axis=1) - primitive["size"][0]
    raise ValueError(primitive["type"])


def union_sdf(points, primitives):
    return np.min([sdf(points, p) for p in primitives], axis=0)


def surface(primitive, count=1500, seed=37):
    rng = np.random.default_rng(seed)
    kind, size = primitive["type"], primitive["size"]
    if kind == "mesh":
        points, _ = trimesh.sample.sample_surface(primitive["mesh"], count, seed=seed)
    elif kind == "box":
        points = rng.uniform(-1, 1, (count, 3)) * size
        areas = np.array([size[1]*size[2], size[0]*size[2], size[0]*size[1]])
        axis = rng.choice(3, count, p=areas/areas.sum())
        points[np.arange(count), axis] = rng.choice([-1, 1], count) * size[axis]
    else:
        points = rng.normal(size=(count, 3))
        points /= np.linalg.norm(points, axis=1)[:, None]
        points *= size[0]
        if kind == "capsule":
            half = size[1]
            points[:, 2] += np.where(points[:, 2] >= 0, half, -half)
            cylinder = rng.random(count) < 2*half/(2*half+2*size[0])
            theta = rng.uniform(0, 2*np.pi, cylinder.sum())
            points[cylinder] = np.column_stack([size[0]*np.cos(theta), size[0]*np.sin(theta),
                                               rng.uniform(-half, half, cylinder.sum())])
    return points @ primitive["rot"].T + primitive["pos"]


def quality(mesh, primitives, seed=190, count=6000):
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    signed = union_sdf(points, primitives)
    proxy = np.concatenate([surface(p, max(400, count//len(primitives)), seed+i) for i, p in enumerate(primitives)])
    proxy = proxy[union_sdf(proxy, primitives) >= -1e-7]
    reverse = distance_to_mesh(mesh_scene(mesh), proxy)
    return dict(mesh_to_proxy_p95_mm=float(np.quantile(np.abs(signed), .95)*1000),
                uncovered_p95_mm=float(np.quantile(np.maximum(signed, 0), .95)*1000),
                mesh_within_2mm=float(np.mean(np.abs(signed) <= .002)),
                proxy_to_mesh_p95_mm=float(np.quantile(reverse, .95)*1000),
                proxy_to_mesh_max_mm=float(reverse.max()*1000), primitives=len(primitives))


def fit_boxes(mesh, slices):
    vertices = np.asarray(mesh.vertices)
    points, _ = trimesh.sample.sample_surface(mesh, 5000, seed=17)
    points = np.concatenate([points, vertices])
    center = points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((points-center).T))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = (points-center) @ axes
    bounds = np.linspace(local[:, 0].min(), local[:, 0].max(), slices+1)
    fitted = []
    for i in range(slices):
        selected = local[(local[:, 0] >= bounds[i]) & (local[:, 0] <= bounds[i+1])]
        if len(selected) < 4:
            continue
        low, high = selected.min(axis=0), selected.max(axis=0)
        low[0], high[0] = bounds[i], bounds[i+1]
        fitted.append(dict(type="box", pos=center+axes @ ((low+high)/2),
                           rot=axes, size=np.maximum((high-low)/2, .0002)))
    return fitted


def fit_capsules(mesh, slices, split_axis=0):
    points, _ = trimesh.sample.sample_surface(mesh, 6000, seed=17)
    center = points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((points-center).T))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = (points-center) @ axes
    bounds = np.linspace(local[:, split_axis].min(), local[:, split_axis].max(), slices+1)
    fitted = []
    for i in range(slices):
        selected = local[(local[:, split_axis] >= bounds[i]) & (local[:, split_axis] <= bounds[i+1])]
        if len(selected) < 20:
            continue
        low, high = np.quantile(selected, [.005, .995], axis=0)
        midpoint = (low+high)/2
        radial = np.linalg.norm(selected[:, 1:]-midpoint[1:], axis=1)
        radius = np.quantile(radial, .80)
        half = max((high[0]-low[0])/2-radius, .0001)
        rot = axes[:, [1, 2, 0]]
        fitted.append(dict(type="capsule", pos=center+axes@midpoint, rot=rot,
                           size=np.array([radius, half, 0.])))
    return fitted


def fit_box_grid(mesh, divisions):
    points, _ = trimesh.sample.sample_surface(mesh, 16000, seed=17)
    points = np.concatenate([points, mesh.vertices])
    center = points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((points-center).T))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = (points-center) @ axes
    bounds = [np.linspace(local[:, j].min(), local[:, j].max(), n+1) for j,n in enumerate(divisions)]
    fitted = []
    for i in range(divisions[0]):
        for j in range(divisions[1]):
            mask = ((local[:, 0]>=bounds[0][i]) & (local[:, 0]<=bounds[0][i+1]) &
                    (local[:, 1]>=bounds[1][j]) & (local[:, 1]<=bounds[1][j+1]))
            selected = local[mask]
            if len(selected) < 4:
                continue
            low, high = selected.min(axis=0), selected.max(axis=0)
            fitted.append(dict(type="box", pos=center+axes@((low+high)/2), rot=axes,
                               size=np.maximum((high-low)/2, .0002)))
    return fitted


def fit_cluster_boxes(mesh, count):
    points, _ = trimesh.sample.sample_surface(mesh, 10000, seed=17)
    groups = KMeans(n_clusters=count, n_init=1, max_iter=50, random_state=17).fit_predict(points)
    fitted = []
    for i in range(count):
        selected = points[groups == i]
        center = selected.mean(axis=0)
        _, axes = np.linalg.eigh(np.cov((selected-center).T))
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
        local = (selected-center)@axes
        low, high = local.min(axis=0), local.max(axis=0)
        fitted.append(dict(type="box", pos=center+axes@((low+high)/2), rot=axes,
                           size=np.maximum((high-low)/2+.0001, .0002)))
    return fitted


def fit_decomposed_boxes(mesh, directory):
    import coacd

    directory.mkdir()
    coacd.set_log_level("warn")
    pieces = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
                             threshold=.03, max_convex_hull=20, resolution=2000,
                             mcts_iterations=50, mcts_nodes=15, seed=17,
                             preprocess_resolution=50, decimate=True, max_ch_vertex=32)
    fitted = []
    for i, (vertices, faces) in enumerate(pieces):
        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        part.export(directory/f"part_{i:02d}.stl")
        transform, extent = trimesh.bounds.oriented_bounds(part)
        inverse = np.linalg.inv(transform)
        fitted.append(dict(type="box", pos=inverse[:3,3], rot=inverse[:3,:3], size=extent/2))
    return fitted


def existing_primitives(model, body_id):
    names = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
             int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule"}
    return [dict(type=names[int(model.geom_type[g])], pos=model.geom_pos[g].copy(),
                 rot=rotation(model.geom_quat[g]), size=model.geom_size[g].copy())
            for g in range(model.ngeom) if model.geom_bodyid[g] == body_id and model.geom_contype[g] == 1]


def fit_convex_parts(parts, output, vertices=48):
    output.mkdir(parents=True, exist_ok=False)
    result=[]
    for i, (_, mesh) in enumerate(parts):
        points=np.asarray(mesh.convex_hull.vertices)
        chosen=[int(np.argmax(np.linalg.norm(points-points.mean(axis=0),axis=1)))]
        distances=np.linalg.norm(points-points[chosen[0]],axis=1)
        for _ in range(min(vertices,len(points))-1):
            chosen.append(int(np.argmax(distances)))
            distances=np.minimum(distances,np.linalg.norm(points-points[chosen[-1]],axis=1))
        reduced=trimesh.convex.convex_hull(points[chosen])
        path=output/f'part_{i:02d}.stl'
        reduced.export(path)
        result.append(dict(type='mesh',mesh=trimesh.load(path,force='mesh'),file=str(path.resolve()),
                           pos=np.zeros(3),rot=np.eye(3),size=np.ones(3)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--clusters", action="store_true")
    parser.add_argument("--coacd", action="store_true")
    parser.add_argument("--palm-hulls", type=Path)
    parser.add_argument("--convex-body", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (args.output/"fit_orca_collision.py").write_bytes(source)
    old = mujoco.MjModel.from_xml_path(str(BASE))
    tree = ET.parse(BASE)
    bodies = {e.get("name"): e for e in tree.iter("body")}
    report = dict(source=str(BASE), source_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
                  method="STL mm to body m; sampled selection of oriented boxes, optional capsules and convex parts",
                  metric_note="Independent sampled bidirectional surface distances, not volume IoU or a real-hardware calibration",
                  builder_sha256=hashlib.sha256(source).hexdigest(),
                  source_meshes={}, bodies={})
    for name, parts in load_parts().items():
        bid = old.body(name).id
        previous = existing_primitives(old, bid)
        if not previous:
            continue
        mesh = trimesh.util.concatenate([m for _, m in parts])
        for path, _ in parts:
            report["source_meshes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = dict(old=quality(mesh, previous), bounds_mm=(mesh.bounds*1000).tolist())
        report["bodies"][name] = entry
        if not args.audit_only:
            candidates = []
            for slices in (1, 2, 3, 4, 6):
                candidate = [p for _, part in parts for p in fit_boxes(part, slices)]
                q = quality(mesh, candidate, seed=90, count=3000)
                objective = max(q["uncovered_p95_mm"], q["proxy_to_mesh_p95_mm"])+.06*len(candidate)
                candidates.append((objective, candidate, q))
            if args.hybrid:
                for slices in (1, 2, 3, 4, 6):
                    for axis in (0, 1):
                        candidate = fit_capsules(mesh, slices, split_axis=axis)
                        q = quality(mesh, candidate, seed=90, count=3000)
                        objective = max(q["uncovered_p95_mm"], q["proxy_to_mesh_p95_mm"])+.06*len(candidate)
                        candidates.append((objective, candidate, q))
            if args.grid:
                for divisions in ((2,2), (3,2), (3,3), (4,3), (4,4), (5,4)):
                    candidate = fit_box_grid(mesh, divisions)
                    q = quality(mesh, candidate, seed=90, count=3000)
                    objective = max(q["uncovered_p95_mm"], q["proxy_to_mesh_p95_mm"])+.06*len(candidate)
                    candidates.append((objective, candidate, q))
            if args.clusters:
                for count in (4, 8, 12, 16, 24):
                    candidate = fit_cluster_boxes(mesh, count)
                    q = quality(mesh, candidate, seed=90, count=3000)
                    objective = max(q["uncovered_p95_mm"], q["proxy_to_mesh_p95_mm"])+.06*len(candidate)
                    candidates.append((objective, candidate, q))
            if args.coacd and ("Carpals" in name or "T-TP-R" in name):
                candidate = fit_decomposed_boxes(mesh, args.output/name)
                q = quality(mesh, candidate, seed=90, count=3000)
                objective = max(q["uncovered_p95_mm"], q["proxy_to_mesh_p95_mm"])+.06*len(candidate)
                candidates.append((objective, candidate, q))
            if args.palm_hulls and "Carpals" in name:
                files = sorted(args.palm_hulls.glob("*.stl"))
                if not files:
                    raise ValueError("No saved palm hulls")
                candidate = [dict(type="mesh", mesh=trimesh.load(p, force="mesh"), file=str(p.resolve()),
                                  pos=np.zeros(3), rot=np.eye(3), size=np.ones(3)) for p in files]
                for p in files:
                    report["source_meshes"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
                q = quality(mesh, candidate, seed=90, count=3000)
                objective = max(q["uncovered_p95_mm"], q["proxy_to_mesh_p95_mm"])+.06*len(candidate)
                candidates.append((objective, candidate, q))
            _, fitted, _ = min(candidates, key=lambda c: c[0])
            if name in args.convex_body:
                fitted=fit_convex_parts(parts,args.output/name)
                for primitive in fitted:
                    p=Path(primitive['file'])
                    report['source_meshes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                entry['local_override']='Per-source-part 48-vertex convex hulls after contact-surface audit'
            entry["new"] = quality(mesh, fitted)
            entry["candidates"] = [dict(objective=c[0], **c[2]) for c in candidates]
            body = bodies[name]
            original = body.findall("geom")
            prefix = original[0].get("name")+"_fit"
            for geom in original:
                if "tip_ball_proxy" in geom.get("name", ""):
                    geom.attrib.update(size="0.0001", contype="0", conaffinity="0", mass="0", group="4")
                else:
                    body.remove(geom)
            # Original masses were inferred from the proxies. Freeze them explicitly
            # so the first comparison changes geometry, not inertial dynamics as well.
            inertial = body.find("inertial")
            if inertial is not None:
                body.remove(inertial)
            ET.SubElement(body, "inertial", pos=text(old.body_ipos[bid]), quat=text(old.body_iquat[bid]),
                          mass=text(old.body_mass[bid]), diaginertia=text(old.body_inertia[bid]))
            original_ids = [old.geom(g.get("name")).id for g in original]
            for i, primitive in enumerate(fitted):
                nearest = min(original_ids, key=lambda g: np.linalg.norm(old.geom_pos[g]-primitive["pos"]))
                friction = old.geom_friction[nearest]
                geom = ET.SubElement(body, "geom", name=f"{prefix}_{i:02d}", type=primitive["type"],
                              pos=text(primitive["pos"]), quat=text(quat(primitive["rot"])),
                              size=text(primitive["size"][:2] if primitive["type"] == "capsule" else primitive["size"]),
                              contype="1", conaffinity="2", group="3",
                              friction=text(friction), rgba="0.15 0.45 1 0.6", mass="0")
                if primitive["type"] == "mesh":
                    mesh_name=f"{prefix}_mesh_{i:02d}"
                    ET.SubElement(tree.getroot().find("asset"), "mesh", name=mesh_name, file=primitive["file"])
                    geom.set("mesh", mesh_name)
                    geom.attrib.pop("size")
        print(name, json.dumps(entry.get("new", entry["old"])), flush=True)
    if not args.audit_only:
        ET.indent(tree)
        path = args.output/"cube.xml"
        tree.write(path, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(path))
        for key in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                    "jnt_pos", "jnt_axis", "jnt_range", "actuator_ctrlrange", "actuator_gainprm"):
            np.testing.assert_allclose(getattr(old, key), getattr(model, key), rtol=1e-8, atol=1e-10)
        report.update(inertia_and_kinematics_match=True, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      collision_geoms=int(np.sum(model.geom_contype == 1)), status="CANDIDATE_NOT_VALIDATED")
    (args.output/"geometry_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
