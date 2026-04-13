import unittest

import numpy as np

from jrso.config import DatasetConfig
from jrso.dataset import SyntheticDatasetGenerator


class SyntheticDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        config = DatasetConfig(
            num_users=18,
            num_items=14,
            embedding_dim=5,
            num_communities=3,
            intra_community_edge_prob=0.32,
            inter_community_edge_prob=0.05,
            interaction_density=0.20,
            k_seeds=3,
            cascade_steps=3,
            num_eval_items=3,
            oracle_simulations=8,
            evaluation_simulations=6,
            seed=5,
        )
        self.dataset = SyntheticDatasetGenerator(config).generate()

    def test_interaction_splits_are_disjoint(self) -> None:
        overlap = self.dataset.train_matrix + self.dataset.val_matrix + self.dataset.test_matrix
        self.assertTrue(np.all(overlap <= 1))
        self.assertTrue(np.array_equal(overlap, self.dataset.interaction_matrix))

    def test_users_with_interactions_keep_train_signal(self) -> None:
        total_positives = self.dataset.interaction_matrix.sum(axis=1)
        train_positives = self.dataset.train_matrix.sum(axis=1)
        active_users = np.flatnonzero(total_positives > 0)
        self.assertTrue(np.all(train_positives[active_users] > 0))

    def test_oracle_targets_cover_evaluation_items(self) -> None:
        evaluation_items = set(map(int, self.dataset.evaluation_items))
        self.assertEqual(evaluation_items, set(self.dataset.oracle_seeds))
        self.assertEqual(evaluation_items, set(self.dataset.oracle_spreads))
        self.assertEqual(evaluation_items, set(self.dataset.oracle_adoption_probabilities))


if __name__ == "__main__":
    unittest.main()
