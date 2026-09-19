import unittest

import numpy as np

from src.experiments.multibench.mosei_purification import _health


class MoseiPurificationTest(unittest.TestCase):
    def test_health_is_finite_for_noncollapsed_target(self):
        health = _health(np.eye(4, dtype=np.float32))
        self.assertGreater(health["variance"], 0.0)
        self.assertTrue(np.isfinite(health["effective_rank"]))


if __name__ == "__main__":
    unittest.main()
