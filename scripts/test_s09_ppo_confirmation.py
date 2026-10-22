"""Selection must stay on development data and respect checkpoint ancestry."""

import csv
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

import freeze_s09_ppo_confirmation as freeze
from retarget_s09_reference import digest
from summarize_s09_ppo_improvement import origin_and_budget


class SelectionTests(unittest.TestCase):
    def make_run(self,root,name,rows,checkpoint=None,start=130000):
        run=root/name
        run.mkdir()
        config=dict(checkpoint=checkpoint,eval_start=start,eval_count=64)
        if checkpoint:
            path=Path(checkpoint)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b"test checkpoint")
            config["checkpoint_sha256"]=digest(path)
        (run/"run_config.json").write_text(json.dumps(config))
        (run/"summary.json").write_text(json.dumps(dict(status="completed")))
        with (run/"metrics.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=["step","success30","hold_sec","max_penetration_m"])
            writer.writeheader()
            for step,success in rows:
                writer.writerow(dict(step=step,success30=success,hold_sec=20,max_penetration_m=.002))
        return run

    def test_discarded_tail_cannot_enter_selected_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            first=self.make_run(root,"first",[(0,.4),(20,.5),(40,.9)])
            second=self.make_run(root,"second",[(0,.5),(20,.6)],str(first/"step_000000020/params.pkl"))
            with mock.patch.object(freeze,"BASE",root), mock.patch.object(freeze,"origin_and_budget",
                    side_effect=lambda run,step:origin_and_budget(run,step,root)):
                chosen=freeze.select_checkpoint(second)
            self.assertEqual(chosen["run"],second)
            self.assertEqual(chosen["budget"],40)
            self.assertEqual(float(chosen["row"]["success30"]),.6)

    def test_test_data_must_not_choose_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.make_run(Path(directory),"test",[(0,.4),(20,.5)],start=150000)
            with self.assertRaises(ValueError):
                freeze.select_checkpoint(run)

    def test_endpoint_is_not_the_best_development_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=self.make_run(root,"complete",[(0,.4),(20,.7),(40,.3)])
            config=json.loads((run/"run_config.json").read_text())
            (run/"run_config.json").write_text(json.dumps(dict(config,steps=40)))
            (run/"summary.json").write_text(json.dumps(dict(status="completed",final_step=40)))
            checkpoint=run/"step_000000040/params.pkl"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"endpoint")
            with mock.patch.object(freeze,"origin_and_budget",return_value=(run,40)):
                actual,budget=freeze.terminal_checkpoint(run)
            self.assertEqual(actual,checkpoint)
            self.assertEqual(budget,40)
            (run/"summary.json").write_text(json.dumps(dict(status="completed",final_step=20)))
            with self.assertRaisesRegex(ValueError,"declared training budget"):
                freeze.terminal_checkpoint(run)
            (run/"summary.json").write_text(json.dumps(dict(status="interrupted",final_step=40)))
            with self.assertRaisesRegex(ValueError,"completed development"):
                freeze.terminal_checkpoint(run)

    def test_zero_training_is_not_a_ppo_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.make_run(Path(directory),"baseline",[(0,.9)])
            with self.assertRaises(ValueError):
                freeze.select_checkpoint(run)

    def test_ties_use_hold_then_depth_then_earlier_step(self):
        row=dict(success30=.5,hold_sec=20,max_penetration_m=.002,step=20)
        self.assertGreater(freeze.selection_key(dict(row,hold_sec=21)),freeze.selection_key(row))
        self.assertGreater(freeze.selection_key(dict(row,max_penetration_m=.001)),freeze.selection_key(row))
        self.assertGreater(freeze.selection_key(dict(row,step=10)),freeze.selection_key(row))

    def test_pair_counts_include_both_gains_and_losses(self):
        from analyze_s09_ppo_confirmation import transitions
        self.assertEqual(transitions([1,1,0,0],[1,0,1,0]),
            dict(retained=1,lost=1,gained=1,neither=1))
        with self.assertRaises(ValueError):
            transitions([1,0],[1])

    def test_plots_keep_selected_and_terminal_models_separate(self):
        from analyze_s09_ppo_confirmation import LABELS, plot_results
        rows=[dict(group=group,success_rate=.4,descriptive_ci95=[.3,.5],
            per_training_seed=[.35,.4,.45] if group!="reference" else [.4],
            secondary_all_depth_success_rate=.38) for group in LABELS]
        differences=[dict(contrast=name+" minus bc",difference=.02,descriptive_ci95=[-.01,.05])
            for name in ("bc_ppo","bc_ppo_last")]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            plot_results(rows,differences,output)
            for name in ("heldout_success.png","paired_gains.png"):
                self.assertGreater((output/name).stat().st_size,1000)

    def test_critic_initialization_is_not_a_new_bc_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            original=root/"bc.pkl"
            original.write_bytes(pickle.dumps((np.zeros(1),{"w":np.ones(2)},np.zeros(2))))
            checkpoint=root/"critic/step_000000000/params.pkl"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(pickle.dumps((np.zeros(1),{"w":np.ones(2)},np.ones(2))))
            metadata=dict(status="completed",actor_and_normalizer_exactly_unchanged=True,
                source_checkpoint=str(original),source_checkpoint_sha256=digest(original),checkpoint_sha256=digest(checkpoint),
                discount=.99,collection_backend="cpu",collection_policy="BC",collection_std=.003,
                collection_seed_range=[140000,140007],training_episodes=6,validation_episodes=2,
                fit_updates=100,fit_batch_size=256,fit_learning_rate=.0003,actual_environment_interactions=1000)
            path=checkpoint.parent.parent/"critic_pretraining.json"
            path.write_text(json.dumps(metadata))
            bc,recipe,cost=freeze.resolve_bc_checkpoint(checkpoint)
            self.assertEqual(bc,original)
            self.assertEqual(cost["actual_environment_interactions"],1000)
            self.assertEqual(recipe["fit_updates"],100)
            checkpoint.write_bytes(pickle.dumps((np.zeros(1),{"w":np.zeros(2)},np.ones(2))))
            metadata["checkpoint_sha256"]=digest(checkpoint)
            path.write_text(json.dumps(metadata))
            with self.assertRaises(AssertionError):
                freeze.resolve_bc_checkpoint(checkpoint)

    def test_reference_cannot_hide_a_learned_correction(self):
        params=(None,{"params":{"hidden_2":{"kernel":np.zeros((128,34)),"bias":np.zeros(34)}}})
        freeze.require_zero_residual(params)
        params[1]["params"]["hidden_2"]["bias"][0]=1e-9
        with self.assertRaises(ValueError):
            freeze.require_zero_residual(params)

    def test_adapter_training_summary_is_not_a_weight_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.make_run(Path(directory),"adapter",[(0,.4),(20,.5)])
            (run/"adapter_config.json").write_text(json.dumps(dict(actor_update_rule="feedback_only")))
            with self.assertRaisesRegex(ValueError,"lacks.*audit"):
                freeze.select_checkpoint(run)
            (run/"adapter_audit.json").write_text(json.dumps(dict(status="failed")))
            with self.assertRaisesRegex(ValueError,"invalid or stale"):
                freeze.select_checkpoint(run)

    def test_failure_gates_overlap_and_do_not_count_successes_as_failures(self):
        from analyze_s09_ppo_confirmation import failure_gate_summary
        def episode(success):
            return dict(success30=success,failure_gates=dict(success30=success,
                invalid_seconds_by_gate=dict(load=2.,height=1.),
                break_events_by_gate=dict(load=3,height=1)))
        rows=failure_gate_summary([episode(False),episode(True)])
        self.assertEqual(len(rows),2)
        self.assertEqual(sum(r["failed_pairs_with_gate"] for r in rows),2)
        self.assertTrue(all(r["failed_policy_episode_pairs"]==1 for r in rows))
        self.assertEqual(next(r for r in rows if r["gate"]=="load")["invalid_seconds_total"],2.)
        self.assertTrue(all(r["invalid_seconds_total"]==0 for r in failure_gate_summary([episode(True)])))
        bad=episode(False)
        bad["failure_gates"]["success30"]=True
        with self.assertRaises(ValueError):
            failure_gate_summary([bad])
        with self.assertRaises(ValueError):
            failure_gate_summary([])


if __name__ == "__main__":
    unittest.main()
