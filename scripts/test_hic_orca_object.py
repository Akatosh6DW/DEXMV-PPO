"""Object-frame regressions: hand and object must share one rigid transform."""

import unittest
from pathlib import Path
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from prepare_hic_orca_object import object_local


class ObjectFrameTests(unittest.TestCase):
    def test_rejected_contact_cap_cannot_enter_matched_training(self):
        from orca_hic53_env import require_full_contact_capacity
        root = Path(__file__).resolve().parents[1] / "artifacts/orca_collision_refit"
        path = root / "O24b_hic53_contact64/cube.xml"
        if not path.exists():
            self.skipTest("Rejected contact-capacity experiment not present")
        with self.assertRaises(ValueError):
            require_full_contact_capacity(mujoco.MjModel.from_xml_path(str(path)))
        require_full_contact_capacity(mujoco.MjModel.from_xml_path(str(root / "O25_hic53_cg128/cube.xml")))

    def test_mount_transform_is_rigid_and_rejects_changed_reference(self):
        from rotate_orca_mount import verify_transform
        root = Path(__file__).resolve().parents[1] / "artifacts/orca_collision_refit/O18b_hic53_top_mount"
        if not root.exists():
            self.skipTest("Mount experiment not present")
        result = verify_transform(root / "cube.xml", root / "reference.npz", root / "mount_transform.json")
        self.assertTrue(result["common_rigid_transform_pass"])
        self.assertLess(result["max_body_position_error_m"], 1e-12)
        with np.load(root / "reference.npz") as z:
            packet = {k: z[k] for k in z.files}
        packet["ctrl"][0, 10] += .01
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "reference.npz"
            np.savez_compressed(altered, **packet)
            with self.assertRaises(ValueError):
                verify_transform(root / "cube.xml", altered, root / "mount_transform.json")

    def test_visual_overlay_resolves_relative_object_mesh(self):
        from inspect_orca_mesh_alignment import build_overlay
        root = Path(__file__).resolve().parents[1] / "artifacts/orca_collision_refit"
        model = root / "O11_hic53_clean_dt5ms/cube.xml"
        if not model.exists():
            self.skipTest("Clean source proxy not present")
        with tempfile.TemporaryDirectory() as directory:
            overlay, report = build_overlay(model, Path(directory) / "overlay.xml")
            self.assertTrue(report["physics_arrays_match"])
            self.assertEqual(overlay.nu, 17)

    def test_finite_only_gate_does_not_authorize_training(self):
        from orca_hic53_env import check_hic53_gates
        root = Path(__file__).resolve().parents[1] / "artifacts/orca_collision_refit"
        gate = root / "O06_hic53_validation/report.json"
        if not gate.exists():
            self.skipTest("Finite-only diagnostic not present")
        with self.assertRaises(ValueError):
            check_hic53_gates(root / "O04_hic53_upright_retarget/cube.xml",
                              root / "O05_hic53_grasp_phase/reference.npz", gate)

    def test_legacy_bc_is_not_silently_reused(self):
        from orca_hic53_env import OrcaHic53Grasp
        with self.assertRaises(ValueError):
            OrcaHic53Grasp(Path("/nonexistent"), "bc", Path("/nonexistent"))

    def test_prepared_object_matches_source_without_changing_hand(self):
        root = Path(__file__).resolve().parents[1]
        new = root / "artifacts/orca_collision_refit/O01_hic53_source_object/cube.xml"
        if not new.exists():
            self.skipTest("Generated source object not present")
        from prepare_hic_orca_object import compiled_vertices, MODEL
        original = mujoco.MjModel.from_xml_path(str(MODEL))
        matched = mujoco.MjModel.from_xml_path(str(new))
        np.testing.assert_allclose(np.ptp(compiled_vertices(matched), axis=0), [.053] * 3, atol=1e-7)
        np.testing.assert_array_equal(original.actuator_ctrlrange, matched.actuator_ctrlrange)
        np.testing.assert_allclose(matched.body("cube").mass, .035)
        for i in range(matched.nbody):
            if i != matched.body("cube").id:
                np.testing.assert_array_equal(original.body_inertia[i], matched.body_inertia[i])
        with self.assertRaises(ValueError):
            compiled_vertices(original)

    def test_reference_task_binding_rejects_scaling(self):
        from orca_hic53_env import task_packet
        root = Path(__file__).resolve().parents[1] / "artifacts/orca_collision_refit"
        path = root / "O05_hic53_grasp_phase/reference.npz"
        if not path.exists():
            self.skipTest("Matched reference not present")
        model = mujoco.MjModel.from_xml_path(str(root / "O04_hic53_upright_retarget/cube.xml"))
        packet = task_packet(model, path)
        self.assertAlmostEqual(packet["lowest_z"], -.0265, places=6)
        with np.load(path) as z:
            fields = {k: z[k] for k in z.files}
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "scaled.npz"
            np.savez_compressed(bad, **dict(fields, spatial_scale=np.array(36 / 53)))
            with self.assertRaises(ValueError):
                task_packet(model, bad)

    def test_common_rigid_transform_preserves_correspondence(self):
        rng = np.random.default_rng(51)
        poses = np.tile(np.eye(4), (6, 1, 1))
        poses[:, :3, :3] = Rotation.random(6, random_state=rng).as_matrix()
        poses[:, :3, 3] = rng.normal(size=(6, 3))
        local = rng.normal(size=(6, 5, 3)) * .05
        world = np.einsum("tij,tkj->tki", poses[:, :3, :3], local) + poses[:, None, :3, 3]
        np.testing.assert_allclose(object_local(world, poses), local, atol=1e-14)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [15, 20, -80], degrees=True).as_matrix()
        transform[:3, 3] = [.3, .1, -.2]
        moved = world @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(object_local(moved, transform @ poses), local, atol=1e-14)

    def test_rejects_scale_reflection_and_unsynchronized_frames(self):
        points = np.zeros((2, 5, 3))
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[:, :3, :3] *= 36 / 53
        with self.assertRaises(ValueError):
            object_local(points, poses)
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[:, 0, 0] = -1
        with self.assertRaises(ValueError):
            object_local(points, poses)
        with self.assertRaises(ValueError):
            object_local(points, poses[:1])


if __name__ == "__main__":
    unittest.main()
