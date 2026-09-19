import unittest

import torch

from src.datasets.synthetic_order import PAIR_KEYS, REGIMES, make_dataset
from src.experiments.synthetic_order.oracle import OracleOrderModel


class SyntheticOrderTest(unittest.TestCase):
    def test_seed_reproducibility_and_balance(self):
        first = make_dataset(REGIMES["s4_mixed"], seed=1)
        second = make_dataset(REGIMES["s4_mixed"], seed=1)
        self.assertTrue(torch.equal(first[0]["x1"], second[0]["x1"]))
        self.assertAlmostEqual(float(first[2]["labels"].float().mean()), 0.5, delta=0.03)

    def test_disabled_orders_are_zero(self):
        for regime, config in REGIMES.items():
            split = make_dataset(config, seed=1)[0]
            first = (split["s1"] + split["s2"] + split["s3"]) / 3**0.5
            expected = config.alpha * first
            pair_scores = {"12_to_3": split["s12"], "13_to_2": split["s13"], "23_to_1": split["s23"]}
            if config.active_pairs:
                expected = expected + config.beta * sum(pair_scores[key] for key in config.active_pairs) / len(config.active_pairs) ** 0.5
            if config.active_triple:
                expected = expected + config.gamma * split["s123"]
            torch.testing.assert_close(split["score"], expected)
            self.assertTrue(torch.equal(split["labels"], (expected > 0).long()))

    def test_pair_mapping_names_are_explicit(self):
        self.assertEqual(PAIR_KEYS, ("12_to_3", "13_to_2", "23_to_1"))
        self.assertEqual(REGIMES["s1_pair12"].active_pairs, ("12_to_3",))

    def test_pair_and_triple_targets_and_normalization(self):
        split = make_dataset(REGIMES["s4_mixed"], seed=1)[0]
        torch.testing.assert_close(split["g12"], split["z1"] * split["z2"])
        torch.testing.assert_close(split["g13"], split["z1"] * split["z3"])
        torch.testing.assert_close(split["g23"], split["z2"] * split["z3"])
        torch.testing.assert_close(split["g123"], split["z1"] * split["z2"] * split["z3"])
        for key in ("s1", "s2", "s3", "s12", "s13", "s23", "s123"):
            self.assertAlmostEqual(float(split[key].var()), 1.0, delta=0.12)

    def test_splits_have_independent_sampling(self):
        train, valid, test = make_dataset(REGIMES["s0_first_order"], seed=1)
        self.assertFalse(torch.equal(train["z1"][0], valid["z1"][0]))
        self.assertFalse(torch.equal(valid["z1"][0], test["z1"][0]))

    def test_disabled_model_branches_are_exactly_zero(self):
        model = OracleOrderModel(64, 32, rank=64, active_pairs=("12_to_3",), active_triple=False)
        output = model(torch.randn(4, 64), torch.randn(4, 64), torch.randn(4, 64))
        self.assertTrue(torch.equal(output["h13"], torch.zeros_like(output["h13"])))
        self.assertTrue(torch.equal(output["h23"], torch.zeros_like(output["h23"])))
        self.assertTrue(torch.equal(output["h123"], torch.zeros_like(output["h123"])))


if __name__ == "__main__":
    unittest.main()
