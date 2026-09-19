import unittest

import numpy as np

from src.experiments.multibench.mosei_identifiability import AdditivePredictor, JointPredictor, _pool, capacity_config, fit_additive_predictor, joint_advantage


class MoseiIdentifiabilityTest(unittest.TestCase):
    def test_capacity_matching_stays_below_one_percent(self):
        config = capacity_config(35, 74, 300)
        self.assertLess(config["parameter_difference_percent"], 1.0)
        self.assertEqual(sum(p.numel() for p in AdditivePredictor(35, 74, 300, config["hidden_dim"]).parameters()), config["additive_parameters"])
        self.assertEqual(sum(p.numel() for p in JointPredictor(35, 74, 300, config["rank"]).parameters()), config["joint_parameters"])

    def test_mapping_advantage_is_joint_minus_additive(self):
        self.assertAlmostEqual(joint_advantage(0.2, 0.35), 0.15)

    def test_pooling_does_not_create_nan(self):
        values = np.zeros((2, 4, 3), dtype=np.float32)
        values[0, 1:] = 1.0
        values[1, 0, 0] = np.nan
        self.assertTrue(np.isfinite(_pool(values)).all())

    def test_predictor_initialization_is_seeded(self):
        rng = np.random.default_rng(7)
        source_a, source_b, target = (rng.normal(size=(4, dim)).astype(np.float32) for dim in (2, 3, 4))
        config = {"hidden_dim": 4}
        kwargs = {"batch_size": 4, "epochs": 0, "patience": 1, "lr": 1e-3, "weight_decay": 0.0}
        first, _ = fit_additive_predictor(source_a, source_b, target, source_a, source_b, target, config, seed=11, **kwargs)
        second, _ = fit_additive_predictor(source_a, source_b, target, source_a, source_b, target, config, seed=11, **kwargs)
        for key, value in first.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), second.state_dict()[key].numpy())


if __name__ == "__main__":
    unittest.main()
