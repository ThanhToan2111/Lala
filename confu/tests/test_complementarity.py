import unittest

import torch
from torch import nn

from src.modules.models.complementarity import (
    AdversarialPredictor,
    EMAStandardizer,
    adversary_loss,
    factor_variance_loss,
    shortcut_hinge_loss,
)
from src.modules.models.synergyformer import LowRankInteraction


class ComplementarityTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.dim = 16
        self.r1 = torch.randn(32, self.dim, requires_grad=True)
        self.r2 = torch.randn(32, self.dim, requires_grad=True)

    def test_adversary_step_no_grad_into_interaction(self) -> None:
        q1 = AdversarialPredictor(self.dim, 32)
        q2 = AdversarialPredictor(self.dim, 32)
        target = torch.randn(32, self.dim, requires_grad=True)
        loss, mse1, mse2 = adversary_loss(q1, q2, self.r1.detach(), self.r2.detach(), target.detach())
        loss.backward()
        self.assertIsNotNone(q1.net[0].weight.grad)
        self.assertIsNotNone(q2.net[0].weight.grad)
        self.assertIsNone(self.r1.grad)
        self.assertIsNone(self.r2.grad)
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(loss))

    def test_generator_step_grad_only_through_target(self) -> None:
        q1 = AdversarialPredictor(self.dim, 32)
        q2 = AdversarialPredictor(self.dim, 32)
        with torch.no_grad():
            pred1 = q1(self.r1)
            pred2 = q2(self.r2)
        target = torch.randn(32, self.dim, requires_grad=True)
        penalty, mse1, mse2 = shortcut_hinge_loss(pred1, pred2, target, tau=2.0)
        penalty.backward()
        # gradient flows into the interaction target ...
        self.assertIsNotNone(target.grad)
        # ... but never into the adversaries
        self.assertIsNone(q1.net[0].weight.grad)
        self.assertIsNone(q2.net[0].weight.grad)
        self.assertGreaterEqual(penalty.item(), 0.0)

    def test_hinge_zero_when_adversary_fails(self) -> None:
        target = torch.zeros(8, self.dim)
        far = torch.full((8, self.dim), 10.0)  # huge mse -> adversary fails
        penalty, mse1, mse2 = shortcut_hinge_loss(far, far, target, tau=0.5)
        self.assertEqual(penalty.item(), 0.0)

    def test_hinge_positive_when_adversary_succeeds(self) -> None:
        target = torch.zeros(8, self.dim)
        near = torch.zeros(8, self.dim)  # zero mse -> shortcut
        penalty, _, _ = shortcut_hinge_loss(near, near, target, tau=0.5)
        self.assertAlmostEqual(penalty.item(), 1.0)

    def test_standardizer_stats_stop_grad(self) -> None:
        standardizer = EMAStandardizer(self.dim)
        x = torch.randn(64, self.dim, requires_grad=True)
        standardizer.update(x)
        t = standardizer.standardize(x)
        t.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(standardizer.mean.requires_grad)
        self.assertFalse(standardizer.var.requires_grad)
        # after many updates stats stay sane
        for _ in range(10):
            standardizer.update(torch.randn(64, self.dim))
        self.assertTrue(torch.isfinite(standardizer.mean).all())
        self.assertTrue((standardizer.var > 0).all())

    def test_factor_variance_loss_detects_collapse(self) -> None:
        collapsed = torch.zeros(64, 8)  # constant factor
        healthy = torch.randn(64, 8)
        loss_collapsed = factor_variance_loss([collapsed, healthy])
        loss_healthy = factor_variance_loss([healthy, torch.randn(64, 8)])
        self.assertGreater(loss_collapsed.item(), loss_healthy.item())
        # gradient reaches the collapsed factor
        collapsed.requires_grad_(True)
        factor_variance_loss([collapsed, healthy]).backward()
        self.assertIsNotNone(collapsed.grad)

    def test_low_rank_project_factors(self) -> None:
        interaction = LowRankInteraction(
            self.dim, 8, 2, normalize_factors=True, factor_bias=False, output_bias=False
        )
        factors = interaction.project_factors(self.r1, self.r2)
        self.assertEqual(len(factors), 2)
        self.assertEqual(factors[0].shape, (32, 8))
        out = interaction(self.r1, self.r2)
        self.assertEqual(out.shape, (32, self.dim))
        self.assertTrue(torch.isfinite(out).all())

    def test_adversary_overfits_frozen_shortcut(self) -> None:
        """Sanity: on a real shortcut (r12 = MLP(r1)), q1 reaches low MSE."""
        torch.manual_seed(1)
        shortcut = nn.Linear(self.dim, self.dim)
        r1 = torch.randn(512, self.dim)
        t = shortcut(r1).detach()
        q1 = AdversarialPredictor(self.dim, 32)
        opt = torch.optim.AdamW(q1.parameters(), lr=1e-3)
        for _ in range(200):
            loss = torch.nn.functional.mse_loss(q1(r1), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        final = torch.nn.functional.mse_loss(q1(r1), t).item()
        initial_t_var = t.var().item()
        self.assertLess(final, 0.1 * initial_t_var)

    def test_batch_size_one(self) -> None:
        q1 = AdversarialPredictor(self.dim, 32)
        standardizer = EMAStandardizer(self.dim)
        x = torch.randn(1, self.dim)
        standardizer.update(x)
        t = standardizer.standardize(x)
        out = q1(x)
        penalty, _, _ = shortcut_hinge_loss(out.detach(), out.detach(), t, tau=0.5)
        self.assertTrue(torch.isfinite(penalty))


if __name__ == "__main__":
    unittest.main()
