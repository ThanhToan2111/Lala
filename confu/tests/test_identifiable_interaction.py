import unittest

import numpy as np
import torch

from src.datasets.identifiable_interaction import REGIMES, make_dataset
from src.experiments.synthetic_interaction.ipib import (
    AdditivePredictor,
    PairPredictor,
    _batch_standardize,
    _interaction_loss,
    _seed_torch,
)
from src.experiments.synthetic_interaction.reliability_audit import reliability_score


class IdentifiableInteractionTest(unittest.TestCase):
    def test_seed_reproducibility_and_components(self):
        first = make_dataset(REGIMES["i2_medium"], 1)
        second = make_dataset(REGIMES["i2_medium"], 1)
        self.assertTrue(torch.equal(first[0]["x3"], second[0]["x3"]))
        expected = first[0]["additive"] + 0.5 * first[0]["joint"] + first[0]["private"]
        torch.testing.assert_close(first[0]["x3"], expected)

    def test_beta_zero_removes_joint_target(self):
        split = make_dataset(REGIMES["i0_no_joint"], 1)[0]
        self.assertEqual(float(split["joint_target"].abs().max()), 0.0)

    def test_task_score_matches_labels(self):
        split = make_dataset(REGIMES["i2_medium"], 1)[0]
        self.assertTrue(torch.equal((split["task_score"] > 0).long(), split["labels"]))

    def test_private_and_source_samples_are_separate(self):
        train, valid, test = make_dataset(REGIMES["i2_medium"], 1)
        self.assertFalse(torch.equal(train["z1"][0], valid["z1"][0]))
        self.assertFalse(torch.equal(valid["u3"][0], test["u3"][0]))

    def test_predictor_capacity_is_matched(self):
        additive = sum(parameter.numel() for parameter in AdditivePredictor().parameters())
        joint = sum(parameter.numel() for parameter in PairPredictor().parameters())
        self.assertLess(abs(joint - additive) / additive, 0.05)

    def test_model_initialization_is_seeded(self):
        _seed_torch(7)
        first = AdditivePredictor().state_dict()
        _seed_torch(7)
        second = AdditivePredictor().state_dict()
        for name in first:
            torch.testing.assert_close(first[name], second[name])

    def test_standardized_distillation_loss_is_finite(self):
        prediction = torch.randn(32, 64, requires_grad=True)
        target = torch.randn(32, 64)
        loss = _interaction_loss(prediction, target, "d2_mse")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_reliability_uses_only_predictor_errors(self):
        target = np.zeros((8, 3))
        additive = np.ones((8, 3))
        joint = np.tile([[0.5, 1.0, 2.0]], (8, 1))
        additive_error, joint_error, reliability = reliability_score(additive, joint, target)
        np.testing.assert_allclose(additive_error, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(joint_error, [0.25, 1.0, 4.0])
        np.testing.assert_allclose(reliability, [0.75, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
