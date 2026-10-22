"""Guard paired development comparisons for the separately reported experts."""

import unittest
from analyze_s09_late_experts import select_runs, transitions


class LateExpertTests(unittest.TestCase):
    def test_budget_and_source_selection(self):
        self.assertEqual(select_runs()[0].name, "E77_bc_late_experts")
        self.assertEqual(select_runs()[3], 163840)
        self.assertEqual(select_runs(long=True)[0].name, "E78_bc_late_experts_long")
        for seed, short, long, control in (
            (14121, "E79_bc_late_experts_s14121", "E104_experts_long_s14121", "E55_bc_budget_restart_s14121"),
            (14123, "E80_bc_late_experts_s14123", "E105_experts_long_s14123", "E56_bc_budget_restart_s14123"),
        ):
            a = select_runs(replica=seed)
            b = select_runs(long_replica=seed)
            self.assertEqual((a[0].name, a[1].name, a[3]), (short, control, 163840))
            self.assertEqual((b[0].name, b[1].name, b[3]), (long, control, 655360))
            self.assertNotEqual(a[2], b[2])
        with self.assertRaises(ValueError):
            select_runs(long=True, replica=14121)
        with self.assertRaises(ValueError):
            select_runs(long_replica=999)

    def test_transitions(self):
        before = [dict(seed=i, selected_reference=i, success30=s)
                  for i,s in enumerate((False, True, True))]
        after = [dict(row, success30=s) for row,s in zip(before, (True, False, True))]
        self.assertEqual(transitions(before, after), dict(gained=1, lost=1))

    def test_reject_unpaired_states_or_routing(self):
        row = dict(seed=1, selected_reference=2, success30=True)
        for before, after in (([], []), ([row], []), ([row], [dict(row, seed=2)]),
                              ([row], [dict(row, selected_reference=3)])):
            with self.assertRaises(ValueError):
                transitions(before, after)


if __name__ == "__main__":
    unittest.main()
