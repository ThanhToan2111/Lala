import unittest

import numpy as np

from src.experiments.multibench.mosei_shortcut_localization import _decision, _mean_std, _r2


class MoseiShortcutLocalizationTest(unittest.TestCase):
    def test_r2_and_summary_helpers(self):
        values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        self.assertAlmostEqual(_r2(values, values), 1.0)
        self.assertAlmostEqual(_mean_std([1.0, 3.0])['mean'], 2.0)

    def test_decision_localizes_mixed_leakage(self):
        summary = {
            "d": {"v_only": {"valid": {"mean": 0.2}}, "a_only": {"valid": {"mean": 0.3}}},
            "h_d2": {"v_only": {"valid": {"mean": 0.31}}, "a_only": {"valid": {"mean": 0.32}}},
        }
        self.assertEqual(_decision(summary), "MIXED")


if __name__ == "__main__":
    unittest.main()
