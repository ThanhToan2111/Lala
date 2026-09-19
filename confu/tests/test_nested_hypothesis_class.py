import unittest

import torch

from src.experiments.hypothesis_class.h0_audit import _seed
from src.experiments.hypothesis_class.nested_joint import (
    NestedJointPredictor,
    ResidualMLP,
    choose_residual_width,
    residual_parameter_count,
)


class FrozenProduct(torch.nn.Module):
    def __init__(self, left_dim, right_dim, target_dim):
        super().__init__()
        self.output = torch.nn.Linear(left_dim + right_dim, target_dim)

    def forward(self, left, right):
        return self.output(torch.cat((left, right), dim=1))


class NestedHypothesisClassTest(unittest.TestCase):
    def test_zero_initialized_nested_equals_product(self):
        product = FrozenProduct(3, 4, 5)
        nested = NestedJointPredictor(product, 3, 4, 5, 2)
        left, right = torch.randn(7, 3), torch.randn(7, 4)
        torch.testing.assert_close(nested(left, right), product(left, right), atol=1e-6, rtol=0.0)

    def test_product_is_frozen_and_residual_has_concatenated_input(self):
        product = FrozenProduct(3, 4, 5)
        nested = NestedJointPredictor(product, 3, 4, 5, 2)
        self.assertTrue(all(not parameter.requires_grad for parameter in nested.product.parameters()))
        self.assertEqual(nested.residual.network[0].in_features, 7)
        self.assertEqual(nested.residual.network[-1].out_features, 5)

    def test_residual_budget_is_fixed_and_below_quarter(self):
        config = choose_residual_width(2048, 32, 300, 53976)
        self.assertLessEqual(config["residual_parameters"], int(0.25 * config["product_parameters"]))
        self.assertEqual(config["residual_parameters"], residual_parameter_count(2048, 32, 300, config["hidden_dim"]))

    def test_residual_output_layer_is_zero_initialized(self):
        model = ResidualMLP(3, 4, 5, 2)
        output = model(torch.randn(4, 3), torch.randn(4, 4))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_residual_initialization_is_deterministic(self):
        _seed(17)
        first = ResidualMLP(3, 4, 5, 2).state_dict()
        _seed(17)
        second = ResidualMLP(3, 4, 5, 2).state_dict()
        for name in first:
            torch.testing.assert_close(first[name], second[name])


if __name__ == "__main__":
    unittest.main()
