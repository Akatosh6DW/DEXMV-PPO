"""Regression checks for experiment binding, pairing and uncertainty reporting."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Aero-Hand/sim_rl/mujoco_playground"))

from analyze_s12_study import hierarchical_interval, wilson
from diagnose_s12_failures import diagnose
from prepare_s12_evaluation import validate_source
from verify_s12_study import physical_retention, residual_effect
from run_s12_job import task_digest
from orca_s12_env import OrcaS12Grasp
from s12_soft import SoftConfig


class ReportingTests(unittest.TestCase):
    def test_physical_retention_still_requires_paired_pad_loads(self):
        cfg = dict(release_sec=5., control_dt=.05, hold_sec=15.)
        trace = dict(time=5.5+np.arange(300)*.05, alive=np.ones(300),
            vertices=np.full((300,4,3),.55),support_force_n=np.zeros(300),
            loads=np.full((300,5),.03))
        self.assertEqual(physical_retention(trace,cfg),(True,15.))
        trace["loads"][:,0] = 0
        self.assertEqual(physical_retention(trace,cfg),(False,0.))
        trace["loads"][:,0] = .03
        trace["loads"][-1,1:] = 0
        success,best = physical_retention(trace,cfg)
        self.assertFalse(success)
        self.assertAlmostEqual(best,14.95)

    def test_requested_residual_can_be_blocked_by_rate_limit(self):
        trace = dict(video_base=np.array([[.99]]),requested_residual=np.array([[.08]]),
            desired_ctrl=np.array([[1.]]),ctrl=np.array([[.1]]))
        result = residual_effect(trace,np.array([-1.]),np.array([1.]))
        self.assertAlmostEqual(result["requested_rms_mean_rad"],.08)
        self.assertAlmostEqual(result["bounded_target_correction_rms_mean_rad"],.01)
        self.assertEqual(result["one_step_command_effect_rms_mean_rad"],0.)

    def test_training_binding_rejects_changed_learning_rate_and_warm_start(self):
        plan = dict(online_steps=51200,jitter=.002,reward="S12-R0",
            train_evaluation_seeds=list(range(10,18)),config={"mesh_version":"test"},model_sha256="test")
        job = dict(seed=1,method="ppo",argv=["python","train.py","--std",".05",
            "--learning-rate",".00003","--evals","6","--physics-workers","4"])
        config = dict(seed=1,method="ppo",steps=51200,jitter=.002,reward_version="S12-R0",
            eval_start=10,eval_count=8,soft_config=plan["config"],model_sha256="test",checkpoint=None,
            bc_phase_only=False,std=.05,learning_rate=.00003,evals=6,physics_workers=4,dataset=None)
        validate_source(config,job,plan)
        with self.assertRaises(ValueError):
            validate_source(dict(config,learning_rate=.0003),job,plan)
        with self.assertRaises(ValueError):
            validate_source(dict(config,checkpoint="partial.pkl"),job,plan)

    def test_task_hash_covers_non_xml_behavior(self):
        cfg = SoftConfig(timestep=.0002,hand_offset_z=.003)
        baseline = OrcaS12Grasp(cfg)
        changed = OrcaS12Grasp(replace(cfg,release_sec=5.2))
        self.assertEqual(baseline.native.sim.xml,changed.native.sim.xml)
        self.assertNotEqual(task_digest(baseline),task_digest(changed))
        raised = OrcaS12Grasp(replace(cfg,hand_offset_z=.006))
        self.assertNotEqual(task_digest(baseline),task_digest(raised))

    def test_fixed_reference_not_replicated_as_independent_sources(self):
        ref = np.array([[0.,1.,0.,1.]])
        learned = np.tile(ref,(3,1))+.25
        self.assertEqual(hierarchical_interval(learned-ref,repeats=1000),[.25,.25])

    def test_seed_variability_not_flattened_away(self):
        a = np.vstack([np.zeros(64),np.ones(64),np.ones(64)])
        low,high = hierarchical_interval(a,repeats=5000)
        self.assertLess(low,.34)
        self.assertEqual(high,1.)

    def test_all_success_still_has_binomial_uncertainty(self):
        low,high = wilson(64,64)
        self.assertGreater(low,.94)
        self.assertLess(low,.95)
        self.assertEqual(high,1.)
        low,high = wilson(0,64)
        self.assertAlmostEqual(low,0.)
        self.assertGreater(high,.05)

    def test_failure_diagnosis_distinguishes_drop_and_quality(self):
        cfg = dict(release_sec=5.,max_strain=.3,max_penetration=.003,max_force=30.)
        z = dict(time=np.array([5.,5.5,24.]),max_principal_strain=np.zeros(3),
            penetration_m=np.zeros(3),peak_force_n=np.zeros(3),
            forbidden_force_n=np.array([.2,0.,0.]),alive=np.ones(3),
            success15=np.array([0.,0.,1.]),hold_sec=np.array([0.,0.,16.]),
            min_volume_ratio=np.ones(3),vertices=np.full((3,4,3),.55))
        self.assertEqual(diagnose(z,cfg)["failure_class"],"passed")
        z["success15"][-1] = 0
        z["forbidden_force_n"][1] = .1
        row = diagnose(z,cfg)
        self.assertEqual(row["failure_class"],"retained_with_quality_violation")
        self.assertEqual(row["first_quality_violation_sec"],5.5)
        self.assertEqual(row["forbidden_contact_violating_control_intervals"],1)
        z["alive"][-1] = 0
        z["vertices"][-1,:,2] = .39
        self.assertEqual(diagnose(z,cfg)["failure_class"],"dropped")
        z["vertices"][-1,:,2] = .55
        z["min_volume_ratio"][-1] = -.06
        row = diagnose(z,cfg)
        self.assertEqual(row["failure_class"],"element_collapse_or_inversion")
        self.assertTrue(row["element_inversion"])
        self.assertFalse(row["dropped"])

    def test_failure_diagnosis_rejects_contradictory_success(self):
        cfg = dict(release_sec=5.,max_strain=.3,max_penetration=.003,max_force=30.)
        z = dict(time=np.array([24.]),max_principal_strain=np.array([.4]),
            penetration_m=np.zeros(1),peak_force_n=np.zeros(1),
            forbidden_force_n=np.zeros(1),alive=np.ones(1),success15=np.ones(1),
            hold_sec=np.array([16.]),min_volume_ratio=np.ones(1),vertices=np.full((1,4,3),.55))
        with self.assertRaises(ValueError):
            diagnose(z,cfg)


if __name__=="__main__":
    unittest.main()
