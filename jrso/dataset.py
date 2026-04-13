from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import DatasetConfig
from .diffusion import celf_select_seeds, estimate_expected_spread


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


@dataclass
class SyntheticDataset:
    """All benchmark tensors and evaluation targets."""

    config: DatasetConfig
    communities: np.ndarray
    social_adjacency: np.ndarray
    influence_probabilities: np.ndarray
    true_user_embeddings: np.ndarray
    true_item_embeddings: np.ndarray
    score_matrix: np.ndarray
    interaction_matrix: np.ndarray
    train_matrix: np.ndarray
    val_matrix: np.ndarray
    test_matrix: np.ndarray
    evaluation_items: np.ndarray
    oracle_seeds: dict[int, list[int]]
    oracle_spreads: dict[int, float]
    oracle_adoption_probabilities: dict[int, np.ndarray]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "dataset.npz",
            communities=self.communities,
            social_adjacency=self.social_adjacency,
            influence_probabilities=self.influence_probabilities,
            true_user_embeddings=self.true_user_embeddings,
            true_item_embeddings=self.true_item_embeddings,
            score_matrix=self.score_matrix,
            interaction_matrix=self.interaction_matrix,
            train_matrix=self.train_matrix,
            val_matrix=self.val_matrix,
            test_matrix=self.test_matrix,
            evaluation_items=self.evaluation_items,
        )
        metadata = {
            "config": self.config.to_dict(),
            "oracle_seeds": {str(k): v for k, v in self.oracle_seeds.items()},
            "oracle_spreads": self.oracle_spreads,
            "oracle_adoption_probabilities": {
                str(k): value.tolist() for k, value in self.oracle_adoption_probabilities.items()
            },
        }
        (output_dir / "dataset_meta.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


class SyntheticDatasetGenerator:
    """Create a corrected synthetic benchmark for JRSO evaluation."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> SyntheticDataset:
        communities = self._generate_communities()
        user_embeddings, item_embeddings, item_topics = self._generate_latent_embeddings(communities)
        social_adjacency = self._generate_social_graph(communities)
        score_matrix = self._generate_preference_scores(
            user_embeddings,
            item_embeddings,
            communities,
            item_topics,
        )
        interaction_matrix = self._sample_interactions(score_matrix)
        train_matrix, val_matrix, test_matrix = self._split_interactions(interaction_matrix)
        influence_probabilities = self._compute_influence_probabilities(
            train_matrix,
            social_adjacency,
            user_embeddings,
        )
        evaluation_items = self._select_evaluation_items(test_matrix)
        oracle_targets = self._compute_oracle_targets(
            evaluation_items,
            score_matrix,
            social_adjacency,
            influence_probabilities,
        )
        return SyntheticDataset(
            config=self.config,
            communities=communities,
            social_adjacency=social_adjacency,
            influence_probabilities=influence_probabilities,
            true_user_embeddings=user_embeddings,
            true_item_embeddings=item_embeddings,
            score_matrix=score_matrix,
            interaction_matrix=interaction_matrix,
            train_matrix=train_matrix,
            val_matrix=val_matrix,
            test_matrix=test_matrix,
            evaluation_items=evaluation_items,
            oracle_seeds=oracle_targets["seeds"],
            oracle_spreads=oracle_targets["spreads"],
            oracle_adoption_probabilities=oracle_targets["adoption_probabilities"],
        )

    def _generate_communities(self) -> np.ndarray:
        base_size = self.config.num_users // self.config.num_communities
        sizes = [base_size] * self.config.num_communities
        sizes[-1] += self.config.num_users - sum(sizes)
        return np.concatenate(
            [np.full(size, community, dtype=np.int64) for community, size in enumerate(sizes)]
        )

    def _generate_latent_embeddings(
        self,
        communities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_communities = self.config.num_communities
        dim = self.config.embedding_dim
        centroids = _row_normalize(self.rng.normal(0.0, 1.0, size=(num_communities, dim)))

        user_noise = self.rng.normal(0.0, 0.35, size=(self.config.num_users, dim))
        user_embeddings = _row_normalize(centroids[communities] + user_noise)

        item_topics = self.rng.integers(0, num_communities, size=self.config.num_items)
        item_noise = self.rng.normal(0.0, 0.45, size=(self.config.num_items, dim))
        item_embeddings = _row_normalize(centroids[item_topics] + item_noise)
        return user_embeddings, item_embeddings, item_topics

    def _generate_social_graph(self, communities: np.ndarray) -> np.ndarray:
        num_users = self.config.num_users
        adjacency = np.zeros((num_users, num_users), dtype=np.float64)
        for u in range(num_users):
            for v in range(u + 1, num_users):
                same = communities[u] == communities[v]
                prob = (
                    self.config.intra_community_edge_prob
                    if same
                    else self.config.inter_community_edge_prob
                )
                if self.rng.random() < prob:
                    adjacency[u, v] = 1.0
                    adjacency[v, u] = 1.0

        for u in range(num_users):
            if adjacency[u].sum() == 0:
                same_pool = np.where((communities == communities[u]) & (np.arange(num_users) != u))[0]
                if same_pool.size == 0:
                    continue
                v = int(self.rng.choice(same_pool))
                adjacency[u, v] = 1.0
                adjacency[v, u] = 1.0
        np.fill_diagonal(adjacency, 0.0)
        return adjacency

    def _generate_preference_scores(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        communities: np.ndarray,
        item_topics: np.ndarray,
    ) -> np.ndarray:
        community_match = (communities[:, None] == item_topics[None, :]).astype(np.float64)
        popularity_bias = self.rng.normal(0.0, 0.15, size=(self.config.num_items,))
        logits = (
            (user_embeddings @ item_embeddings.T) / np.sqrt(self.config.embedding_dim)
            + 0.65 * community_match
            + popularity_bias[None, :]
        )
        return np.clip(_sigmoid(logits), 1e-4, 1.0 - 1e-4)

    def _sample_interactions(self, score_matrix: np.ndarray) -> np.ndarray:
        scaled = np.clip(
            score_matrix * self.config.interaction_density * 2.0,
            0.0,
            0.92,
        )
        interactions = (self.rng.random(score_matrix.shape) < scaled).astype(np.int8)

        for user_idx in range(self.config.num_users):
            if interactions[user_idx].sum() == 0:
                top_item = int(np.argmax(score_matrix[user_idx]))
                interactions[user_idx, top_item] = 1
        return interactions

    def _split_interactions(
        self,
        interaction_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train = np.zeros_like(interaction_matrix, dtype=np.int8)
        val = np.zeros_like(interaction_matrix, dtype=np.int8)
        test = np.zeros_like(interaction_matrix, dtype=np.int8)

        for user_idx in range(interaction_matrix.shape[0]):
            positives = np.flatnonzero(interaction_matrix[user_idx])
            if positives.size == 0:
                continue
            shuffled = self.rng.permutation(positives)
            if positives.size == 1:
                train[user_idx, shuffled[0]] = 1
                continue

            test_count = max(1, int(round(positives.size * self.config.test_ratio)))
            val_count = max(1, int(round(positives.size * self.config.val_ratio))) if positives.size >= 4 else 0
            if test_count + val_count >= positives.size:
                overflow = test_count + val_count - positives.size + 1
                if val_count >= overflow:
                    val_count -= overflow
                else:
                    test_count = max(1, test_count - (overflow - val_count))
                    val_count = 0

            test_items = shuffled[:test_count]
            val_items = shuffled[test_count : test_count + val_count]
            train_items = shuffled[test_count + val_count :]

            if train_items.size == 0:
                train_items = shuffled[-1:]
                test_items = shuffled[:-1]
                val_items = np.array([], dtype=np.int64)

            train[user_idx, train_items] = 1
            val[user_idx, val_items] = 1
            test[user_idx, test_items] = 1

        return train, val, test

    def _compute_influence_probabilities(
        self,
        train_matrix: np.ndarray,
        social_adjacency: np.ndarray,
        user_embeddings: np.ndarray,
    ) -> np.ndarray:
        num_users = train_matrix.shape[0]
        influence = np.zeros((num_users, num_users), dtype=np.float64)
        item_sets = [set(np.flatnonzero(train_matrix[user_idx])) for user_idx in range(num_users)]
        similarity = _sigmoid((user_embeddings @ user_embeddings.T) / np.sqrt(self.config.embedding_dim))
        for source in range(num_users):
            for target in np.flatnonzero(social_adjacency[source]):
                intersection = len(item_sets[source] & item_sets[target])
                union = len(item_sets[source] | item_sets[target]) or 1
                jaccard = intersection / union
                prob = 0.04 + 0.38 * jaccard + 0.20 * similarity[source, target]
                influence[source, target] = float(np.clip(prob, 0.02, 0.60))
        np.fill_diagonal(influence, 0.0)
        return influence

    def _select_evaluation_items(self, test_matrix: np.ndarray) -> np.ndarray:
        item_popularity = test_matrix.sum(axis=0)
        viable = np.flatnonzero(item_popularity > 0)
        if viable.size == 0:
            viable = np.arange(min(self.config.num_items, self.config.num_eval_items))
        if viable.size <= self.config.num_eval_items:
            return viable.astype(np.int64)
        top = viable[np.argsort(item_popularity[viable])[::-1]]
        return top[: self.config.num_eval_items].astype(np.int64)

    def _compute_oracle_targets(
        self,
        evaluation_items: np.ndarray,
        score_matrix: np.ndarray,
        social_adjacency: np.ndarray,
        influence_probabilities: np.ndarray,
    ) -> dict[str, Any]:
        oracle_seeds: dict[int, list[int]] = {}
        oracle_spreads: dict[int, float] = {}
        oracle_adoption_probabilities: dict[int, np.ndarray] = {}
        for item_idx in evaluation_items:
            item_scores = score_matrix[:, item_idx]
            seeds = celf_select_seeds(
                item_scores=item_scores,
                social_adjacency=social_adjacency,
                influence_probabilities=influence_probabilities,
                k=self.config.k_seeds,
                cascade_steps=self.config.cascade_steps,
                influence_scale=self.config.influence_scale,
                num_trials=self.config.oracle_simulations,
                rng_seed=self.config.seed + int(item_idx),
            )
            spread, adoption_probabilities = estimate_expected_spread(
                item_scores=item_scores,
                social_adjacency=social_adjacency,
                influence_probabilities=influence_probabilities,
                seeds=seeds,
                cascade_steps=self.config.cascade_steps,
                influence_scale=self.config.influence_scale,
                num_trials=self.config.oracle_simulations,
                rng_seed=self.config.seed + 1_000 + int(item_idx),
            )
            oracle_seeds[int(item_idx)] = list(map(int, seeds))
            oracle_spreads[int(item_idx)] = float(spread)
            oracle_adoption_probabilities[int(item_idx)] = adoption_probabilities
        return {
            "seeds": oracle_seeds,
            "spreads": oracle_spreads,
            "adoption_probabilities": oracle_adoption_probabilities,
        }
