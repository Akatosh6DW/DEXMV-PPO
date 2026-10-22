import unittest

from make_s09_checkpoint_portfolio import choose_sources


class PortfolioSelectionTests(unittest.TestCase):
    def test_counts_and_earliest_ties(self):
        self.assertEqual(choose_sources([[9,6,7,3,0,1],[9,4,3,9,0,4],[9,3,8,6,5,5]]), [0,0,2,1,2,2])

    def test_invalid_counts_rejected(self):
        for counts in ([], [[0]*6]*2, [[-1]*6]*3, [[.5]*6]*3):
            with self.assertRaises(ValueError):
                choose_sources(counts)


if __name__ == "__main__":
    unittest.main()
