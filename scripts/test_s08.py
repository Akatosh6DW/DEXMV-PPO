"""Regression checks for the new wrap contract and unchanged residual control."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import mujoco
import numpy as np

from orca_native_physics import STATE_SPEC
from orca_s08_env import OrcaS08Grasp
from probe_s08_wrap import contacts

INPUT=Path(__file__).resolve().parents[1]/"artifacts/orca_benchmark_suite/S08/inputs"


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=OrcaS08Grasp(INPUT/"model.xml","residual",INPUT/"reference.npz",height_weight=5)

    def test_nominal_physics_unchanged(self):
        expected=mujoco.MjModel.from_xml_path(str(INPUT/"model.xml"))
        for field in ("body_mass","body_inertia","geom_friction","geom_size","actuator_gainprm"):
            np.testing.assert_array_equal(getattr(expected,field),getattr(self.env.mj_model,field))
        self.assertEqual(self.env.mj_model.neq,0)
        self.assertEqual(self.env.n_substeps,10)

    def test_palm_allowed_nonhand_forbidden(self):
        m=self.env.mj_model
        for i in range(m.ngeom):
            if m.geom(i).name.startswith("orca_palm_"):
                self.assertNotIn(i,self.env._forbidden_gids)
        self.assertIn(m.geom("cube_support").id,self.env._forbidden_gids)

    def test_cylinder_axis_includes_geom_local_rotation(self):
        m=self.env.mj_model
        d=mujoco.MjData(m)
        body=m.body("cube")
        address=m.jnt_qposadr[body.jntadr[0]]+3
        for degrees in (-10.,0.,10.):
            angle=np.deg2rad(degrees)/2
            yaw=np.array([np.cos(angle),0.,0.,np.sin(angle)])
            quat=np.zeros(4)
            mujoco.mju_mulQuat(quat,yaw,body.quat)
            d.qpos[address:address+4]=quat
            mujoco.mj_forward(m,d)
            axis=d.geom_xmat[m.geom("cube").id].reshape(3,3)[:,2]
            np.testing.assert_allclose(axis,[0.,0.,1.],atol=1e-12)

    def test_reset_gate_and_observation(self):
        states=jax.jit(jax.vmap(self.env.reset))(jax.vmap(jax.random.PRNGKey)(np.arange(85000,85008)))
        self.assertLessEqual(float(states.info["initial_penetration"].max()),.001)
        self.assertEqual(states.obs["state"].shape,(8,123))
        self.assertEqual(states.obs["privileged_state"].shape,(8,140))
        self.assertTrue(np.isfinite(states.obs["state"]).all())
        self.assertTrue(np.all(states.obs["state"][:,94]==0))

    def test_minimum_contact_load_over_substeps(self):
        native=self.env.native
        d=mujoco.MjData(self.env.mj_model)
        mujoco.mj_forward(self.env.mj_model,d)
        native.loads[:]=np.inf
        native.pad_audit[:2]=1
        sequence=[np.array([1,1,1,1,0,0.]),np.array([0,1,1,1,0,0.])]
        for loads in sequence:
            with patch("orca_s08_env.contacts",return_value=(loads,False,1.)):
                native.sample_contacts(d)
        self.assertEqual(native.loads[0],0)
        self.assertEqual(native.pad_audit[0],0)

    def test_native_reference_step_parity(self):
        env=self.env
        state=jax.jit(env.reset)(jax.random.PRNGKey(85000))
        state=jax.device_get(state)
        model=env.mj_model
        d=mujoco.MjData(model)
        words=np.ascontiguousarray(state.data.integration_words,dtype=np.uint32).view(np.float64)
        mujoco.mj_setState(model,d,words,STATE_SPEC)
        step=jax.jit(env.step)
        for k in range(80):
            # Replay each callback from its exact double-precision incoming state.
            # Long independent rollouts amplify CPU/GPU float32 rounding at contact.
            words=np.ascontiguousarray(state.data.integration_words,dtype=np.uint32).view(np.float64)
            mujoco.mj_setState(model,d,words,STATE_SPEC)
            previous_ctrl=np.asarray(state.data.ctrl)
            state=step(state,np.zeros(17,np.float32))
            d.ctrl[:]=np.asarray(state.data.ctrl)
            d.mocap_pos[:]=np.asarray(state.data.mocap_pos)
            self.assertLessEqual(float(np.max(np.abs(d.ctrl-previous_ctrl))),.100001)
            np.testing.assert_allclose(state.info["video_base"],env.reference_ctrl[k],atol=1e-7)
            loads=np.full(6,np.inf)
            for _ in range(10):
                mujoco.mj_step(model,d)
                loads=np.minimum(loads,contacts(model,d)[0])
            np.testing.assert_allclose(state.data.qpos,d.qpos,atol=1e-7,rtol=1e-6)
            np.testing.assert_allclose(state.data.wrap_loads,loads,atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(state.info["requested_residual"],0,atol=1e-7)

    def test_interventions_apply_to_final_model(self):
        for benchmark in ("B3_friction","B4_mass"):
            env=OrcaS08Grasp(INPUT/"model.xml","residual",INPUT/"reference.npz",benchmark=benchmark)
            if benchmark=="B4_mass":
                self.assertAlmostEqual(env.mj_model.body("cube").mass[0],.07)
            else:
                mask=env.mj_model.geom_contype!=0
                np.testing.assert_allclose(env.mj_model.geom_friction[mask,0],self.env.mj_model.geom_friction[mask,0]*.6)

    def classify(self,name,normal,reversed_pair=False):
        m=self.env.mj_model
        gid,cube=m.geom(name).id,m.geom("cube").id
        pair=(cube,gid) if reversed_pair else (gid,cube)
        c=SimpleNamespace(geom=pair,geom1=pair[0],geom2=pair[1],dist=-.0001,
                          frame=np.r_[np.asarray(normal)*(-1 if reversed_pair else 1),np.zeros(6)])
        d=SimpleNamespace(contact=[c],ncon=1,xmat=np.tile(np.eye(3).ravel(),(m.nbody,1)),
                          geom_xmat=np.tile(np.eye(3).ravel(),(m.ngeom,1)))
        def force(_m,_d,_i,out):
            out[:]=0;out[0]=1
        with patch("probe_s08_wrap.mujoco.mj_contactForce",side_effect=force):
            return contacts(m,d)

    def test_surface_normals_and_reversed_contact_order(self):
        for reverse in (False,True):
            good,bad,_=self.classify("orca_palm_proxy_fit_00",[0,1,0],reverse)
            self.assertEqual(good[0],1);self.assertFalse(bad)
            good,bad,_=self.classify("orca_thumb_tip_proxy_fit_02",[-1,0,0],reverse)
            self.assertEqual(good[1],1);self.assertFalse(bad)
        for normal in ([0,-1,0],[0,0,1]):
            good,_,_=self.classify("orca_palm_proxy_fit_00",normal)
            self.assertEqual(good.sum(),0)

    def test_assistance_is_distinct_from_qualifying_load_and_support(self):
        good,bad,_=self.classify("orca_thumb_mid_proxy_fit_00",[-1,0,0])
        self.assertEqual(good.sum(),0);self.assertFalse(bad)
        _,bad,_=self.classify("cube_support",[0,0,1])
        self.assertTrue(bad)


if __name__=="__main__":
    unittest.main()
