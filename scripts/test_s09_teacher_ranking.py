"""Preselect the same candidate pool without consulting long outcomes."""

import unittest
from probe_s09_teacher_ranking import shortlist, select_cases


class TeacherRankingTests(unittest.TestCase):
    def test_unique_short_candidates_and_zero(self):
        history=[dict(cost=9.,delta=[0.]*10),dict(cost=1.,delta=[.02]*10),
                 dict(cost=1.,delta=[.02]*10),dict(cost=2.,delta=[-.02]*10)]
        self.assertEqual(shortlist(history,2),[1,3,0])
        self.assertEqual(shortlist(history,1),[1,0])
        self.assertEqual(shortlist(history,8),[1,3,0])

    def test_invalid_pool(self):
        for history in ([],[dict(cost=float("nan"),delta=[0.]*10)],
                        [dict(cost=0.,delta=[.09]*10)]):
            with self.assertRaises(ValueError):
                shortlist(history)

    def test_query_subset_stays_training_only(self):
        self.assertEqual(select_cases([1,2,3],[2,3],False),[2,3])
        self.assertEqual(select_cases([1,2,3],None,True),[1])
        for requested in ([],[2,2],[4]):
            with self.assertRaises(ValueError):
                select_cases([1,2,3],requested,False)


if __name__=="__main__":
    unittest.main()
