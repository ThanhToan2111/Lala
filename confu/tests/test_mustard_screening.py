import unittest

import numpy as np

from src.experiments.multibench.mustard_screening import MAPPINGS, _final_type, _g1_status, _pool, load_mustard
from src.experiments.multibench.mosei_task_headroom import parameter_counts


class MustardScreeningTest(unittest.TestCase):
    def test_mapping_directions(self):
        self.assertEqual(MAPPINGS["VA_to_T"], ("vision", "audio", "text"))
        self.assertEqual(MAPPINGS["VT_to_A"], ("vision", "text", "audio"))
        self.assertEqual(MAPPINGS["AT_to_V"], ("audio", "text", "vision"))

    def test_masked_pool_ignores_padding(self):
        value = np.array([[[1.0, 0.0], [0.0, 0.0], [3.0, 0.0]]], dtype=np.float32)
        np.testing.assert_allclose(_pool(value), [[2.0, 0.0]])

    def test_task_capacity_is_controlled(self):
        audit = parameter_counts(371, 81)
        self.assertLess(audit["mismatch_vs_additive_percent"]["joint_product"], 1.0)
        self.assertLess(audit["mismatch_vs_additive_percent"]["concat_mlp"], 1.0)

    def test_g1_requires_all_screening_seeds(self):
        records = [{"metrics": {"valid": {"joint_advantage": value}}} for value in (0.02, 0.03, 0.009)]
        self.assertEqual(_g1_status(records), "WEAK_UNSTABLE")

    def test_two_gate_type_matrix(self):
        self.assertEqual(_final_type("FAIL", "FAIL"), "TYPE_I")
        self.assertEqual(_final_type("FAIL", "WEAK_UNSTABLE"), "INCONCLUSIVE")
        self.assertEqual(_final_type("PASS", "FAIL"), "TYPE_II")


if __name__ == "__main__":
    unittest.main()
