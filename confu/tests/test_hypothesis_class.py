import unittest

import numpy as np
import torch

from src.experiments.hypothesis_class.h0_audit import (
    GenericJointMLP,
    _decision,
    _seed,
    choose_mlp_width,
    mlp_parameter_count,
)


class HypothesisClassTest(unittest.TestCase):
    def test_generic_mlp_concatenates_two_inputs_and_matches_target(self):
        model = GenericJointMLP(3, 4, 5, 6)
        self.assertEqual(tuple(model(torch.randn(2, 3), torch.randn(2, 4)).shape), (2, 5))
        self.assertEqual(len(model.network), 3)

    def test_capacity_matching_stays_below_maximum(self):
        audit = choose_mlp_width(2048, 32, 300, 54240)
        self.assertLess(audit["mismatch_vs_reference_percent"], 5.0)
        self.assertEqual(audit["mlp_parameters"], mlp_parameter_count(2048, 32, 300, audit["hidden_dim"]))

    def test_seed_initialization_is_deterministic(self):
        _seed(13)
        first = GenericJointMLP(3, 4, 5, 6).state_dict()
        _seed(13)
        second = GenericJointMLP(3, 4, 5, 6).state_dict()
        for name in first:
            torch.testing.assert_close(first[name], second[name])

    def test_h0_decision_uses_validation_candidate(self):
        row = {
            "summary": {
                "valid": {"J_mlp": {"per_seed": [0.011, 0.012, 0.013]}},
            }
        }
        decision, candidates, reason = _decision({"VA_to_T": row})
        self.assertEqual(decision, "PRODUCT_CLASS_INSUFFICIENT")
        self.assertTrue(candidates["VA_to_T"])
        self.assertTrue(reason)

    def test_test_values_cannot_create_candidate(self):
        row = {
            "summary": {
                "valid": {"J_mlp": {"per_seed": [-0.001, -0.002, -0.003]}},
                "test": {"J_mlp": {"per_seed": [0.2, 0.2, 0.2]}},
            }
        }
        decision, candidates, reason = _decision({"VA_to_T": row})
        self.assertNotEqual(decision, "PRODUCT_CLASS_INSUFFICIENT")
        self.assertFalse(candidates["VA_to_T"])
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
