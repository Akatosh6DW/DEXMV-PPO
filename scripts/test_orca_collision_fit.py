"""Geometry units, distances, frame transforms and primitive surface checks."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from fit_orca_collision import (distance_to_mesh, fit_cluster_boxes, fit_convex_parts, load_parts,
                                mesh_scene, quat, rotation, sdf, surface, union_sdf)


class CollisionFitTests(unittest.TestCase):
    def test_saved_convex_part_preserves_units_and_frame(self):
        mesh=trimesh.creation.box(extents=[.02,.03,.05])
        mesh.apply_translation([.01,.02,.03])
        with tempfile.TemporaryDirectory() as directory:
            parts=fit_convex_parts([(Path('source.stl'),mesh)],Path(directory)/'hulls')
            self.assertEqual(len(parts),1)
            self.assertLessEqual(len(parts[0]['mesh'].vertices),48)
            np.testing.assert_allclose(parts[0]['mesh'].bounds,mesh.bounds,atol=1e-8)

    def test_convex_mesh_signed_distance_and_transform(self):
        primitive=dict(type='mesh',mesh=trimesh.creation.box(extents=[2,2,2]),
                       pos=np.array([.1,.2,.3]),rot=Rotation.from_euler('xyz',[.2,.4,.6]).as_matrix(),size=np.zeros(3))
        local=np.array([[0,0,0],[2,0,0],[1,0,0]])
        world=local@primitive['rot'].T+primitive['pos']
        np.testing.assert_allclose(sdf(world,primitive),[-1,1,0],atol=1e-6)
        np.testing.assert_allclose(sdf(surface(primitive),primitive),0,atol=1e-6)

    def test_quaternion_frame_roundtrip(self):
        matrix = Rotation.from_euler("xyz", [.4, -.8, 1.1]).as_matrix()
        np.testing.assert_allclose(rotation(quat(matrix)), matrix, atol=1e-12)

    def test_signed_primitive_distances(self):
        base = dict(pos=np.zeros(3), rot=np.eye(3))
        box = dict(**base, type="box", size=np.array([1., 2., 3.]))
        np.testing.assert_allclose(sdf(np.array([[0,0,0], [2,2,3], [2,3,4]]), box), [-1,1,np.sqrt(3)])
        capsule = dict(**base, type="capsule", size=np.array([1.,2.,0.]))
        np.testing.assert_allclose(sdf(np.array([[0,0,0], [0,0,4], [2,0,0]]), capsule), [-1,1,1])

    def test_sampled_surfaces_follow_transforms(self):
        for kind,size in [("box", [.01,.02,.03]), ("capsule", [.01,.02,0]), ("sphere", [.01,0,0])]:
            primitive = dict(type=kind, size=np.array(size), pos=np.array([.1,.2,.3]),
                             rot=Rotation.from_euler("xyz", [.2,.4,.6]).as_matrix())
            np.testing.assert_allclose(sdf(surface(primitive), primitive), 0, atol=1e-12)

    def test_triangle_distance_is_metric_and_unsigned(self):
        scene = mesh_scene(trimesh.creation.box(extents=[2,2,2]))
        np.testing.assert_allclose(distance_to_mesh(scene, [[0,0,0], [2,0,0], [1,0,0]]), [1,1,0], atol=1e-6)

    def test_loaded_finger_units_are_meters(self):
        parts = load_parts()["right_I-FingerTipAssembly_ec49c16c"]
        mesh = trimesh.util.concatenate([m for _,m in parts])
        self.assertLess(mesh.extents.max(), .06)
        self.assertGreater(mesh.extents.max(), .035)

    def test_cluster_fit_covers_independent_box_samples(self):
        mesh = trimesh.creation.box(extents=[.02,.03,.05])
        fitted = fit_cluster_boxes(mesh, 8)
        points,_ = trimesh.sample.sample_surface(mesh, 3000, seed=91)
        self.assertLess(float(np.quantile(np.maximum(union_sdf(points, fitted), 0), .95)), .001)


if __name__ == "__main__":
    unittest.main()
