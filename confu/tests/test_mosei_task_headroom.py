import unittest

import torch

from src.experiments.multibench.mosei_task_headroom import (
    AdditiveTaskPredictor,
    ConcatMLPTaskPredictor,
    JointTaskPredictor,
    _decision,
    parameter_counts,
)


class MoseiTaskHeadroomTest(unittest.TestCase):
    def test_architectures_shapes_and_capacity(self):
        vision, audio = torch.randn(5, 713), torch.randn(5, 74)
        models = (
            AdditiveTaskPredictor(713, 74),
            JointTaskPredictor(713, 74),
            ConcatMLPTaskPredictor(787),
        )
        for model in models:
            self.assertEqual(model(vision, audio).shape, (5,))
        audit = parameter_counts(713, 74)
        self.assertLess(max(audit["mismatch_vs_additive_percent"][name] for name in ("joint_product", "concat_mlp")), 1.0)

    def test_additive_branches_are_separate(self):
        model = AdditiveTaskPredictor(3, 2)
        vision, audio = torch.randn(4, 3), torch.randn(4, 2)
        delta = model(vision, audio + 1.0) - model(vision, audio)
        expected = (model.audio(audio + 1.0) - model.audio(audio)).squeeze(1)
        self.assertTrue(torch.allclose(delta, expected))

    def test_decision_does_not_open_on_inconsistent_signs(self):
        metric = lambda values: {"mean": 0.01, "std": 0.1, "ci95": [-0.1, 0.1], "per_seed": values}
        paired = {
            "joint_minus_additive": {"valid": {name: metric([1, -1, 1, -1, 1]) for name in ("accuracy", "macro_f1", "auc", "bce_improvement")}},
            "mlp_minus_additive": {"valid": {name: metric([1, -1, 1, -1, 1]) for name in ("accuracy", "macro_f1", "auc", "bce_improvement")}},
        }
        self.assertEqual(_decision(paired), "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
