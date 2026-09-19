import unittest

import numpy as np
import torch

from src.experiments.multibench.meld_screening import (
    EMOTION_LABELS,
    MAPPINGS,
    NUM_CLASSES,
    AdditiveTaskPredictor,
    JointTaskPredictor,
    _build,
    _final_type,
    _g2_status,
    _g1_status,
    _masked_mean,
    _seed,
    _standardize,
    parameter_counts,
)


class MeldScreeningTest(unittest.TestCase):
    def test_mapping_directions(self):
        self.assertEqual(MAPPINGS["VA_to_T"], ("vision", "audio", "text"))
        self.assertEqual(MAPPINGS["VT_to_A"], ("vision", "text", "audio"))
        self.assertEqual(MAPPINGS["AT_to_V"], ("audio", "text", "vision"))

    def test_masked_mean_ignores_padding(self):
        value = np.array([[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]], dtype=np.float32)
        np.testing.assert_allclose(_masked_mean(value, 2), [2.0, 0.0])

    def test_capacity_is_controlled(self):
        audit = parameter_counts(2048, 32)
        self.assertLess(audit["mismatch_vs_additive_percent"]["joint_product"], 1.0)
        self.assertLess(audit["mismatch_vs_additive_percent"]["concat_mlp"], 1.0)

    def test_multiclass_heads_emit_seven_logits(self):
        left = torch.randn(3, 8)
        right = torch.randn(3, 5)
        for name in ("linear", "additive", "joint_product", "concat_mlp"):
            self.assertEqual(tuple(_build(name, 8, 5)(left, right).shape), (3, NUM_CLASSES))

    def test_additive_branches_are_isolated(self):
        model = AdditiveTaskPredictor(8, 5).eval()
        left_a, left_b, right_a, right_b = (torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 5), torch.randn(3, 5))
        delta_a = model(left_a, right_a) - model(left_b, right_a)
        delta_b = model(left_a, right_b) - model(left_b, right_b)
        torch.testing.assert_close(delta_a, delta_b)

    def test_joint_product_matches_explicit_formula(self):
        model = JointTaskPredictor(8, 5).eval()
        left, right = torch.randn(3, 8), torch.randn(3, 5)
        left_factor, right_factor = model.left(left), model.right(right)
        expected = model.output(torch.cat((left_factor, right_factor, left_factor * right_factor), dim=1))
        torch.testing.assert_close(model(left, right), expected)

    def test_train_only_normalization(self):
        splits = {
            "train": {name: np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32) for name in ("vision", "audio", "text")},
            "valid": {name: np.array([[5.0, 7.0]], dtype=np.float32) for name in ("vision", "audio", "text")},
            "test": {name: np.array([[1.0, 5.0]], dtype=np.float32) for name in ("vision", "audio", "text")},
        }
        normalized = _standardize(splits)
        np.testing.assert_allclose(normalized["train"]["vision"].mean(0), [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(normalized["valid"]["vision"], [[3.0, 3.0]], atol=1e-6)

    def test_gate_ignores_test_sign(self):
        headroom = {
            "valid": {"macro_f1": {"mean": -0.1, "per_seed": [-0.1, -0.1, -0.1]}, "nll_improvement": {"mean": -0.1}},
            "test": {"macro_f1": {"mean": 0.3, "per_seed": [0.3, 0.3, 0.3]}, "nll_improvement": {"mean": 0.3}},
        }
        self.assertEqual(_g2_status(headroom), "FAIL")

    def test_seed_initialization_is_deterministic(self):
        _seed(17)
        first = AdditiveTaskPredictor(8, 5).state_dict()
        _seed(17)
        second = AdditiveTaskPredictor(8, 5).state_dict()
        for name in first:
            torch.testing.assert_close(first[name], second[name])

    def test_gate_and_label_contract(self):
        self.assertEqual(len(EMOTION_LABELS), 7)
        records = [{"metrics": {"valid": {"joint_advantage": value}}} for value in (0.02, 0.03, 0.009)]
        self.assertEqual(_g1_status(records), "WEAK_UNSTABLE")
        self.assertEqual(_final_type("FAIL", "FAIL"), "TYPE_I")
        self.assertEqual(_final_type("PASS", "FAIL"), "TYPE_II")


if __name__ == "__main__":
    unittest.main()
