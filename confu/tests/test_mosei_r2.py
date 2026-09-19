import unittest

import torch

from src.experiments.multibench.mosei_r2 import CanonicalInteraction


class MoseiR2Test(unittest.TestCase):
    def test_canonical_interaction_shape_and_finiteness(self):
        model = CanonicalInteraction(713, 74, 300)
        output = model(torch.randn(4, 713), torch.randn(4, 74))
        self.assertEqual(output.shape, (4, 300))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
