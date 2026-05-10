from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .torch_config import TorchDatasetConfig
from .torch_models import _build_lightgcn_adjacency, _csr_to_torch_sparse


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _row_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _stratified_eval_items(popularity: np.ndarray, num_eval_items: int) -> np.ndarray:
    candidates = np.flatnonzero(popularity > 0)
    if candidates.size <= num_eval_items:
        return candidates.astype(np.int64)
    ordered = candidates[np.argsort(popularity[candidates])]
    positions = np.linspace(0, ordered.size - 1, num=num_eval_items)
    chosen_positions = np.unique(np.round(positions).astype(np.int64))
    if chosen_positions.size < num_eval_items:
        remaining = np.setdiff1d(np.arange(ordered.size, dtype=np.int64), chosen_positions, assume_unique=False)
        needed = min(num_eval_items - chosen_positions.size, remaining.size)
        if needed > 0:
            chosen_positions = np.sort(np.concatenate([chosen_positions, remaining[-needed:]]))
    selected = ordered[chosen_positions[:num_eval_items]]
    return selected[np.argsort(popularity[selected])[::-1]].astype(np.int64)


def _weighted_pagerank(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    num_nodes: int,
    alpha: float = 0.85,
    iterations: int = 20,
) -> np.ndarray:
    src, dst = edge_index
    out_weight = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(out_weight, src, edge_weight)
    rank = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float32)
    base = (1.0 - alpha) / num_nodes
    for _ in range(iterations):
        contribution = rank[src] * edge_weight / np.clip(out_weight[src], 1e-6, None)
        updated = np.full(num_nodes, base, dtype=np.float32)
        np.add.at(updated, dst, alpha * contribution)
        rank = updated
    return rank


@dataclass
class SparseSyntheticDataset:
    """Sparse synthetic benchmark suitable for GPU training."""

    config: TorchDatasetConfig
    communities: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    centrality: np.ndarray
    true_user_embeddings: np.ndarray
    true_item_embeddings: np.ndarray
    score_matrix: np.ndarray
    train_matrix: np.ndarray
    val_matrix: np.ndarray
    test_matrix: np.ndarray
    train_pairs: np.ndarray
    evaluation_items: np.ndarray
    item_popularity: np.ndarray

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "torch_dataset.npz",
            communities=self.communities,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            centrality=self.centrality,
            true_user_embeddings=self.true_user_embeddings,
            true_item_embeddings=self.true_item_embeddings,
            score_matrix=self.score_matrix,
            train_matrix=self.train_matrix.astype(np.uint8),
            val_matrix=self.val_matrix.astype(np.uint8),
            test_matrix=self.test_matrix.astype(np.uint8),
            train_pairs=self.train_pairs,
            evaluation_items=self.evaluation_items,
            item_popularity=self.item_popularity,
        )
        metadata = {
            "config": self.config.to_dict(),
            "num_edges": int(self.edge_index.shape[1]),
            "num_train_pairs": int(self.train_pairs.shape[0]),
        }
        (output_dir / "torch_dataset_meta.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def to_torch(self, device: torch.device) -> dict[str, torch.Tensor]:
        adjacency = _build_lightgcn_adjacency(
            train_matrix=self.train_matrix,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            num_users=int(self.config.num_users),
            num_items=int(self.config.num_items),
            social_scale=0.5,
        )
        sparse_adj = _csr_to_torch_sparse(adjacency).to(device)
        return {
            "edge_index": torch.as_tensor(self.edge_index, dtype=torch.long, device=device),
            "edge_weight": torch.as_tensor(self.edge_weight, dtype=torch.float32, device=device),
            "centrality": torch.as_tensor(self.centrality, dtype=torch.float32, device=device),
            "lightgcn_adj": sparse_adj,
        }


class SparseSyntheticDatasetGenerator:
    """Generate a sparse large-scale synthetic benchmark."""

    def __init__(self, config: TorchDatasetConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> SparseSyntheticDataset:
        communities = self._generate_communities()
        user_embeddings, item_embeddings, item_topics = self._generate_latent_embeddings(communities)
        edge_index = self._generate_sparse_social_graph(communities)
        score_matrix = self._generate_scores(user_embeddings, item_embeddings, communities, item_topics)
        interaction_matrix = self._generate_interactions(score_matrix)
        train_matrix, val_matrix, test_matrix = self._split_interactions(interaction_matrix)
        edge_weight = self._compute_edge_weights(edge_index, train_matrix, user_embeddings, communities)
        centrality = _weighted_pagerank(edge_index, edge_weight, self.config.num_users)
        evaluation_items = self._select_evaluation_items(test_matrix)
        train_pairs = np.argwhere(train_matrix).astype(np.int64)
        item_popularity = train_matrix.sum(axis=0).astype(np.int64)
        return SparseSyntheticDataset(
            config=self.config,
            communities=communities,
            edge_index=edge_index,
            edge_weight=edge_weight,
            centrality=centrality,
            true_user_embeddings=user_embeddings,
            true_item_embeddings=item_embeddings,
            score_matrix=score_matrix,
            train_matrix=train_matrix,
            val_matrix=val_matrix,
            test_matrix=test_matrix,
            train_pairs=train_pairs,
            evaluation_items=evaluation_items,
            item_popularity=item_popularity,
        )

    def _generate_communities(self) -> np.ndarray:
        base = self.config.num_users // self.config.num_communities
        sizes = [base] * self.config.num_communities
        sizes[-1] += self.config.num_users - sum(sizes)
        return np.concatenate(
            [np.full(size, community, dtype=np.int64) for community, size in enumerate(sizes)]
        )

    def _generate_latent_embeddings(
        self,
        communities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_communities = self.config.num_communities
        latent_dim = self.config.latent_dim
        centroids = _row_normalize(
            self.rng.normal(0.0, 1.0, size=(num_communities, latent_dim)).astype(np.float32)
        )
        user_noise = self.rng.normal(0.0, 0.30, size=(self.config.num_users, latent_dim)).astype(np.float32)
        item_topics = self.rng.integers(0, num_communities, size=self.config.num_items, endpoint=False)
        item_noise = self.rng.normal(0.0, 0.40, size=(self.config.num_items, latent_dim)).astype(np.float32)
        user_embeddings = _row_normalize(centroids[communities] + user_noise).astype(np.float32)
        item_embeddings = _row_normalize(centroids[item_topics] + item_noise).astype(np.float32)
        return user_embeddings, item_embeddings, item_topics.astype(np.int64)

    def _generate_sparse_social_graph(self, communities: np.ndarray) -> np.ndarray:
        community_members = [
            np.flatnonzero(communities == community) for community in range(self.config.num_communities)
        ]
        other_members = [
            np.flatnonzero(communities != community) for community in range(self.config.num_communities)
        ]
        undirected_edges: set[tuple[int, int]] = set()
        for user_idx in range(self.config.num_users):
            community = int(communities[user_idx])
            same_pool = community_members[community][community_members[community] != user_idx]
            if same_pool.size > 0:
                intra_degree = min(
                    same_pool.size,
                    max(1, int(self.rng.poisson(self.config.avg_intra_degree))),
                )
                intra_neighbors = self.rng.choice(same_pool, size=intra_degree, replace=False)
            else:
                intra_neighbors = np.array([], dtype=np.int64)

            cross_pool = other_members[community]
            if cross_pool.size > 0:
                inter_degree = min(
                    cross_pool.size,
                    max(1, int(self.rng.poisson(self.config.avg_inter_degree))),
                )
                cross_neighbors = self.rng.choice(cross_pool, size=inter_degree, replace=False)
            else:
                cross_neighbors = np.array([], dtype=np.int64)

            for neighbor in np.concatenate([intra_neighbors, cross_neighbors]):
                source, target = sorted((int(user_idx), int(neighbor)))
                if source != target:
                    undirected_edges.add((source, target))

        directed: list[tuple[int, int]] = []
        for source, target in undirected_edges:
            directed.append((source, target))
            directed.append((target, source))
        directed.sort()
        return np.asarray(directed, dtype=np.int64).T

    def _generate_scores(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        communities: np.ndarray,
        item_topics: np.ndarray,
    ) -> np.ndarray:
        score_matrix = np.empty((self.config.num_users, self.config.num_items), dtype=np.float32)
        item_bias = self.rng.normal(0.0, 0.25, size=self.config.num_items).astype(np.float32)
        chunk_size = 1_024
        for start in range(0, self.config.num_users, chunk_size):
            end = min(self.config.num_users, start + chunk_size)
            community_match = (communities[start:end, None] == item_topics[None, :]).astype(np.float32)
            logits = (
                (user_embeddings[start:end] @ item_embeddings.T) / np.sqrt(self.config.latent_dim)
                + 0.70 * community_match
                + item_bias[None, :]
            )
            score_matrix[start:end] = np.clip(_sigmoid(logits), 1e-4, 1.0 - 1e-4).astype(np.float32)
        return score_matrix

    def _generate_interactions(self, score_matrix: np.ndarray) -> np.ndarray:
        num_users, num_items = score_matrix.shape
        interactions = np.zeros((num_users, num_items), dtype=bool)
        mean_interactions = max(
            self.config.min_interactions_per_user,
            int(round(num_items * self.config.interaction_density)),
        )
        for user_idx in range(num_users):
            num_interactions = int(self.rng.poisson(mean_interactions))
            num_interactions = int(
                np.clip(
                    num_interactions,
                    self.config.min_interactions_per_user,
                    self.config.max_interactions_per_user,
                )
            )
            weights = score_matrix[user_idx].astype(np.float64)
            weights /= weights.sum()
            chosen = self.rng.choice(num_items, size=num_interactions, replace=False, p=weights)
            interactions[user_idx, chosen] = True
        return interactions

    def _split_interactions(
        self,
        interaction_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train = np.zeros_like(interaction_matrix, dtype=bool)
        val = np.zeros_like(interaction_matrix, dtype=bool)
        test = np.zeros_like(interaction_matrix, dtype=bool)
        for user_idx in range(interaction_matrix.shape[0]):
            positives = np.flatnonzero(interaction_matrix[user_idx])
            shuffled = self.rng.permutation(positives)
            if positives.size <= 2:
                train[user_idx, shuffled] = True
                continue
            test_count = max(1, int(round(positives.size * self.config.test_ratio)))
            val_count = max(1, int(round(positives.size * self.config.val_ratio)))
            if test_count + val_count >= positives.size:
                val_count = max(0, positives.size - test_count - 1)
            test_items = shuffled[:test_count]
            val_items = shuffled[test_count : test_count + val_count]
            train_items = shuffled[test_count + val_count :]
            if train_items.size == 0:
                train_items = shuffled[-1:]
                test_items = shuffled[:-1]
                val_items = np.array([], dtype=np.int64)
            train[user_idx, train_items] = True
            val[user_idx, val_items] = True
            test[user_idx, test_items] = True
        return train, val, test

    def _compute_edge_weights(
        self,
        edge_index: np.ndarray,
        train_matrix: np.ndarray,
        user_embeddings: np.ndarray,
        communities: np.ndarray,
    ) -> np.ndarray:
        src, dst = edge_index
        train_sets = [set(np.flatnonzero(train_matrix[user_idx])) for user_idx in range(train_matrix.shape[0])]
        similarity = _sigmoid(
            np.sum(user_embeddings[src] * user_embeddings[dst], axis=1) / np.sqrt(self.config.latent_dim)
        )
        same_community = (communities[src] == communities[dst]).astype(np.float32)
        jaccards = np.zeros(src.shape[0], dtype=np.float32)
        for edge_idx, (source, target) in enumerate(zip(src, dst)):
            intersection = len(train_sets[source] & train_sets[target])
            union = len(train_sets[source] | train_sets[target]) or 1
            jaccards[edge_idx] = intersection / union
        weights = 0.03 + 0.34 * jaccards + 0.22 * similarity + 0.05 * same_community
        return np.clip(weights, 0.02, 0.60).astype(np.float32)

    def _select_evaluation_items(self, test_matrix: np.ndarray) -> np.ndarray:
        popularity = test_matrix.sum(axis=0)
        if not bool(np.any(popularity > 0)):
            return np.arange(min(self.config.num_eval_items, self.config.num_items), dtype=np.int64)
        return _stratified_eval_items(popularity, self.config.num_eval_items)
