"""Continuation budgets include ancestors but not discarded training tails."""

import json
from pathlib import Path
import tempfile
import unittest

from summarize_s09_ppo_improvement import origin_and_budget


class AncestryTests(unittest.TestCase):
    def test_selected_ancestor_step_and_external_bc_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            first,second=base/"first",base/"second"
            first.mkdir()
            second.mkdir()
            (first/"run_config.json").write_text(json.dumps({"checkpoint":"/outside/step_000000000/params.pkl"}))
            (second/"run_config.json").write_text(json.dumps({"checkpoint":str(first/"step_000020480/params.pkl")}))
            origin,budget=origin_and_budget(second,163840,base)
            self.assertEqual(origin,first)
            self.assertEqual(budget,184320)

    def test_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/"run"
            run.mkdir()
            (run/"run_config.json").write_text(json.dumps({"checkpoint":str(run/"step_000020480/params.pkl")}))
            with self.assertRaises(ValueError):
                origin_and_budget(run,10,Path(folder))


if __name__ == "__main__":
    unittest.main()
