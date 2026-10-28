"""S12 topology, contact ownership, action semantics and physical checks."""

from dataclasses import replace
from collections import Counter
from pathlib import Path
import unittest

import mujoco
import numpy as np

from s12_soft import SoftConfig, SoftGrasp, tetra_cylinder, tetra_metrics
from render_s12_soft import load_snapshot


class SoftTests(unittest.TestCase):
    def test_render_reconstructs_cached_geometry_after_fast_motion(self):
        sim = SoftGrasp(SoftConfig(timestep=.0002))
        sim.reset()
        sim.data.qvel[-1] = .7
        mujoco.mj_step(sim.model, sim.data)
        trace = {key:np.array([value]) for key,value in dict(
            time=sim.data.time, qpos=sim.data.qpos.astype(np.float32),
            qvel=sim.data.qvel.astype(np.float32),ctrl=sim.data.ctrl.copy(),
            mocap=sim.data.mocap_pos.copy(),mocap_quat=sim.data.mocap_quat.copy(),
            vertices=sim.data.flexvert_xpos.astype(np.float32)).items()}
        renderer_data = mujoco.MjData(sim.model)
        renderer_data.qpos[:] = trace["qpos"][0]
        mujoco.mj_forward(sim.model, renderer_data)
        self.assertGreater(np.max(abs(renderer_data.flexvert_xpos-trace["vertices"][0])),1e-5)
        self.assertLess(load_snapshot(sim.model,renderer_data,trace,0),1e-7)

    def test_boundary_is_closed_without_internal_cracks(self):
        for sectors in (8,12,24):
            cfg = SoftConfig(sectors=sectors)
            v,tet = tetra_cylinder(cfg)
            faces = Counter(tuple(sorted(cell[list(ids)])) for cell in tet
                            for ids in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)))
            self.assertEqual(set(faces.values()), {1,2})
            shell = [face for face,count in faces.items() if count==1]
            self.assertEqual(len(shell), 2*sectors*cfg.layers)
            edges = Counter(tuple(sorted((face[i],face[j]))) for face in shell
                            for i,j in ((0,1),(0,2),(1,2)))
            self.assertEqual(set(edges.values()), {2})
        sim = SoftGrasp(SoftConfig(timestep=.0002))
        v = sim.vertices
        tri = sim.model.flex_shell.reshape(-1,3)
        normals = np.cross(v[tri[:,1]]-v[tri[:,0]], v[tri[:,2]]-v[tri[:,0]])
        self.assertTrue(np.all(np.sum(normals*(v[tri].mean(1)-v.mean(0)), axis=1)>0))

    def test_volume_and_rotational_objectivity(self):
        cfg = SoftConfig()
        v, tet = tetra_cylinder(cfg)
        edges = (v[tet[:, 1:]]-v[tet[:, :1]]).transpose(0, 2, 1)
        volume = np.abs(np.linalg.det(edges)).sum()/6
        expected = .5*cfg.sectors*np.sin(2*np.pi/cfg.sectors)*cfg.radius**2*2*cfg.half_length
        self.assertAlmostEqual(volume, expected, places=12)
        rotation = np.array([[0, -1., 0], [1., 0, 0], [0, 0, 1.]])
        strain, jac, _ = tetra_metrics(v@rotation.T+[1, 2, 3], tet, np.linalg.inv(edges))
        self.assertLess(strain, 1e-12)
        self.assertAlmostEqual(jac, 1., places=12)

    def test_object_has_only_deformable_contacts(self):
        sim = SoftGrasp(SoftConfig(timestep=.0002))
        self.assertEqual(sim.model.nflex, 1)
        self.assertEqual(sim.model.nu, 17)
        self.assertEqual(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "cube"), -1)
        self.assertTrue(np.all(sim.model.actuator_trnid[:, 0] < sim.model.njnt))
        self.assertAlmostEqual(float(sim.model.body_mass[sim.soft_bodies].sum()), .035)
        self.assertTrue(np.all(sim.data.xfrc_applied==0))
        self.assertEqual(sim.model.neq, 1)
        self.assertEqual(sim.model.equality(0).name, "s10_positioner")

    def test_reset_does_not_accumulate_offsets(self):
        sim = SoftGrasp(SoftConfig(timestep=.0002))
        sim.reset(offset=[.001, -.002, .001])
        first = sim.data.qpos.copy()
        sim.reset(offset=[.001, -.002, .001])
        np.testing.assert_array_equal(first, sim.data.qpos)
        sim.step(np.zeros(17))
        self.assertFalse(np.any(sim.data.warning.number))
        self.assertTrue(np.all(sim.data.xfrc_applied==0))

    def test_vectorized_audit_matches_contact_force_api(self):
        sim = SoftGrasp(SoftConfig(timestep=.0002))
        sim.reset()
        for _ in range(6):
            sim.step(np.zeros(17))
            fast, slow = sim.audit(), sim.audit(fast=False)
            for key in fast:
                np.testing.assert_allclose(fast[key], slow[key], atol=1e-12)

    def test_loaded_v2_grasp_audit(self):
        trace = Path(__file__).resolve().parents[1]/"artifacts/orca_benchmark_suite/S12/V2_P08_raise_hand3mm/trajectory.npz"
        if not trace.exists():
            self.skipTest("Requires the completed V2 physical probe")
        sim = SoftGrasp(SoftConfig(timestep=.0002))
        sim.data = mujoco.MjData(sim.model)
        with np.load(trace) as z:
            index = int(np.argmin(np.abs(z["time"]-10)))
            load_snapshot(sim.model,sim.data,z,index)
        fast,slow = sim.audit(),sim.audit(fast=False)
        self.assertTrue(np.isfinite(sim.data.efc_force).all())
        self.assertGreater(fast["loads"].sum(),.1)
        for key in fast:
            np.testing.assert_allclose(fast[key],slow[key],atol=1e-12)


if __name__ == "__main__":
    unittest.main()
