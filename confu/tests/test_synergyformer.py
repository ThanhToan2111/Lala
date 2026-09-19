import unittest

import torch

from src.modules.models.synergyformer import (
    GatedComposition,
    LowRankInteraction,
    SynergyFormer,
    conditional_utility_loss,
    covariance_loss,
    cross_covariance_loss,
    dependence_ranking_loss,
    redundancy_loss,
    variance_loss,
)
from src.experiments.av_mnist.synergyformer import checkpoint_epoch


class SynergyFormerTest(unittest.TestCase):
    def test_checkpoint_epoch(self) -> None:
        self.assertEqual(checkpoint_epoch("best-epoch=02.ckpt"), 2)
        self.assertIsNone(checkpoint_epoch("best.ckpt"))

    def test_bimodal_interaction_and_composition(self) -> None:
        interaction = LowRankInteraction(
            8, 4, 2, normalize_factors=True, factor_bias=False, output_bias=False
        )
        composition = GatedComposition(8)
        left, right = torch.randn(5, 8, requires_grad=True), torch.randn(5, 8, requires_grad=True)
        pair = interaction(left, right)
        fused = composition(left, right, pair)

        self.assertEqual(pair.shape, (5, 8))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertAlmostEqual(composition.gate.item(), 0.5)
        fused.square().mean().backward()
        self.assertIsNotNone(left.grad)
        self.assertIsNotNone(right.grad)
        self.assertIsNotNone(composition.gate_logit.grad)

        zero_gate = composition(left.detach(), right.detach(), pair.detach(), gate=0.0)
        shuffled_zero_gate = composition(
            left.detach(), right.detach(), pair.detach().flip(0), gate=0.0
        )
        self.assertTrue(torch.allclose(zero_gate, shuffled_zero_gate))
        self.assertFalse(
            torch.allclose(
                composition(left.detach(), right.detach(), pair.detach(), gate=1.0),
                composition(left.detach(), right.detach(), pair.detach().flip(0), gate=1.0),
            )
        )

        one = interaction(torch.randn(1, 8), torch.randn(1, 8))
        self.assertTrue(torch.isfinite(one).all())

    def test_variance_and_cross_covariance_losses(self) -> None:
        torch.manual_seed(0)
        collapsed = torch.zeros(2048, 8)
        healthy = torch.randn(2048, 8)
        self.assertGreater(variance_loss(collapsed).item(), variance_loss(healthy).item())
        self.assertGreater(cross_covariance_loss(healthy, healthy).item(), 0.5)
        self.assertLess(cross_covariance_loss(healthy, torch.randn_like(healthy)).item(), 0.02)
        duplicated = torch.randn(2048, 1).repeat(1, 8)
        self.assertGreater(covariance_loss(duplicated).item(), covariance_loss(healthy).item())
        self.assertTrue(torch.isfinite(cross_covariance_loss(torch.randn(1, 8), torch.randn(1, 8))))
        self.assertTrue(torch.isfinite(covariance_loss(torch.randn(1, 8))))

    def test_conditional_utility_loss_and_stop_gradient(self) -> None:
        better = conditional_utility_loss(torch.tensor([0.1]), torch.tensor([1.0]), margin=0.1)
        self.assertEqual(better.item(), 0.0)

        full = torch.tensor([1.0, 0.8], requires_grad=True)
        base = torch.tensor([0.2, 0.4], requires_grad=True)
        loss = conditional_utility_loss(full, base)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(full.grad)
        self.assertIsNone(base.grad)

    def test_dependence_ranking_loss(self) -> None:
        labels = torch.tensor([0, 1])
        positive = torch.tensor([[4.0, 0.0], [0.0, 4.0]], requires_grad=True)
        easy_negative = torch.zeros(2, 2, requires_grad=True)
        hard_negative = positive.detach().clone().requires_grad_()
        easy = dependence_ranking_loss(positive, (easy_negative,), labels, margin=0.1)
        hard = dependence_ranking_loss(positive, (hard_negative,), labels, margin=0.1)
        self.assertLess(easy.item(), hard.item())
        hard.backward()
        self.assertIsNotNone(positive.grad)
        self.assertIsNotNone(hard_negative.grad)

    def test_shapes_gradients_missing_and_finite(self) -> None:
        model = SynergyFormer((2, 3, 4), dim=8, rank=4, num_heads=2)
        inputs = (torch.randn(5, 7, 2), torch.randn(5, 3, 3), torch.randn(5, 4))
        masks = (torch.ones(5, 7, dtype=torch.bool), torch.ones(5, 3, dtype=torch.bool), None)
        outputs = model(*inputs, masks=masks)

        for name in ("r1", "r2", "r3", "r12", "r13", "r23", "r123", "z_global"):
            self.assertEqual(outputs[name].shape, (5, 8))
            self.assertTrue(torch.isfinite(outputs[name]).all())
        loss = outputs["z_global"].square().mean() + redundancy_loss(outputs)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

        missing = model(inputs[0], inputs[1], None, masks=masks)
        self.assertFalse(missing["availability"][:, 2].any())
        for name in ("r3", "r13", "r23", "r123"):
            self.assertTrue(torch.equal(missing[name], torch.zeros_like(missing[name])))

        one = model(torch.randn(1, 2), torch.randn(1, 3), torch.randn(1, 4))
        self.assertTrue(torch.isfinite(redundancy_loss(one)))


if __name__ == "__main__":
    unittest.main()
