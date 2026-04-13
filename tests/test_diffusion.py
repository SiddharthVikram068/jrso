import unittest

import numpy as np

from jrso.diffusion import adoption_probability, continuous_seed_probabilities, estimate_expected_spread


class DiffusionTests(unittest.TestCase):
    def test_adoption_probability_increases_with_influence(self) -> None:
        low = adoption_probability(base_preference=0.35, influence_sum=0.0, influence_scale=1.2)
        high = adoption_probability(base_preference=0.35, influence_sum=1.8, influence_scale=1.2)
        self.assertLess(low, high)

    def test_continuous_seed_probabilities_are_bounded(self) -> None:
        logits = np.array([1.5, -0.5, 0.0, 2.0], dtype=np.float64)
        probabilities = np.asarray(continuous_seed_probabilities(logits, budget=3))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_expected_spread_respects_seed_count(self) -> None:
        item_scores = np.array([0.9, 0.7, 0.5, 0.2], dtype=np.float64)
        adjacency = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        influence = np.array(
            [
                [0.0, 0.4, 0.0, 0.0],
                [0.4, 0.0, 0.4, 0.0],
                [0.0, 0.4, 0.0, 0.4],
                [0.0, 0.0, 0.4, 0.0],
            ]
        )
        single, _ = estimate_expected_spread(
            item_scores=item_scores,
            social_adjacency=adjacency,
            influence_probabilities=influence,
            seeds=[0],
            cascade_steps=3,
            influence_scale=1.1,
            num_trials=24,
            rng_seed=13,
        )
        double, _ = estimate_expected_spread(
            item_scores=item_scores,
            social_adjacency=adjacency,
            influence_probabilities=influence,
            seeds=[0, 1],
            cascade_steps=3,
            influence_scale=1.1,
            num_trials=24,
            rng_seed=13,
        )
        self.assertGreaterEqual(double, single)


if __name__ == "__main__":
    unittest.main()
