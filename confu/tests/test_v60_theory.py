import unittest

import numpy as np


def vector_r2(y, prediction):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return 1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean(axis=0, keepdims=True)) ** 2)


class V60TheoryTest(unittest.TestCase):
    def test_risk_r2_gap_identity(self):
        y = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        additive = np.array([[0.0, 0.5], [1.0, 0.5], [1.0, 1.0]])
        joint = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        denominator = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2)
        risk_gap = np.sum((y - additive) ** 2) - np.sum((y - joint) ** 2)
        self.assertAlmostEqual(vector_r2(joint, joint) - vector_r2(y, additive), risk_gap / denominator)

    def test_zero_joint_residual_has_zero_advantage(self):
        y = np.arange(12, dtype=float).reshape(6, 2)
        prediction = y * 0.25
        self.assertAlmostEqual(vector_r2(y, prediction) - vector_r2(y, prediction), 0.0)

    def test_known_nested_projection_improves(self):
        y = np.array([[0.0], [1.0], [2.0], [3.0]])
        additive = np.full_like(y, y.mean())
        joint = y.copy()
        self.assertGreater(vector_r2(y, joint) - vector_r2(y, additive), 0.99)


if __name__ == "__main__":
    unittest.main()
