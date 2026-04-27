"""Real-data Ciao dataset adapter for the JRSO torch backend.

The Ciao corpus (Tang et al., 2012) contains:

* ``trustnetwork.mat`` -> ``trustnetwork`` : ``(E, 2)`` directed (truster, trustee)
* ``rating.mat``       -> ``rating``       : ``(N, >=4)`` columns
                                              ``[userId, productId, categoryId, rating, ...]``

This module produces a :class:`CiaoDataset` whose attribute layout is **exactly
the same** as :class:`jrso.torch_dataset.SparseSyntheticDataset`, so every torch
model (RecSys-only / JRSO / JRSO_second / Two-Stage / IM-only) and every
helper in :mod:`jrso.torch_experiment` consumes it with no model-side changes.

Key choices made here, with justifications:

1. **K-core filter.** Ciao raw is 7,375 users * 105,114 items but most items
   have <5 ratings. We iteratively drop users below ``user_core`` and items
   below ``item_core`` until stable. After 5-core the active matrix is small
   enough to fit a dense ``(num_users, num_items)`` score matrix in memory.

2. **Per-user leave-out splits.** Standard JRSO/SARS protocol: 70/10/20 train
   /val/test per user, *not* a row-level split (a row-level split would put
   whole users in val/test which makes recommendation impossible).

3. **Edge probabilities = Goyal-style Bernoulli MLE.** For a directed trust
   edge ``u -> v``,
       ``p(u -> v) = (|A_train(u) cap A_train(v)| + s) / (|A_train(u)| + s/floor)``
   smoothed with ``s = edge_prob_smoothing`` and bounded into
   ``[edge_prob_floor, edge_prob_ceiling]``. This is a standard,
   purely-data-driven IC influence estimator.

4. **Score matrix proxy.** The simulator needs *some* per-(user, item) base
   adoption probability. Building this from any trained model would bias the
   benchmark, so we use a **model-free** proxy that depends only on training
   data:
       ``base(u, i) = sigmoid(b + a*log_pop_z(i) + c*neighbor_train_rate(u, i)
                              + d*category_match(u, i))``
   This is what the IC simulator uses *uniformly for every model*, so any
   model that beats the others is winning on its own merits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import scipy.io
import scipy.sparse as sp
import torch

from .ciao_config import CiaoDatasetConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _row_sparse_csr(rows: np.ndarray, cols: np.ndarray, num_rows: int, num_cols: int) -> sp.csr_matrix:
    data = np.ones(rows.shape[0], dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))


def _weighted_pagerank(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    num_nodes: int,
    alpha: float = 0.85,
    iterations: int = 30,
) -> np.ndarray:
    src, dst = edge_index
    out_weight = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(out_weight, src, edge_weight.astype(np.float32))
    rank = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float32)
    base = (1.0 - alpha) / num_nodes
    for _ in range(iterations):
        contribution = rank[src] * edge_weight / np.clip(out_weight[src], 1e-6, None)
        updated = np.full(num_nodes, base, dtype=np.float32)
        np.add.at(updated, dst, alpha * contribution)
        rank = updated
    rank = rank / (rank.sum() + 1e-12)
    return rank.astype(np.float32)


@dataclass
class CiaoDataset:
    """Real-data Ciao benchmark with the same surface as ``SparseSyntheticDataset``."""

    config: CiaoDatasetConfig
    edge_index: np.ndarray
    edge_weight: np.ndarray
    centrality: np.ndarray
    train_matrix: np.ndarray
    val_matrix: np.ndarray
    test_matrix: np.ndarray
    train_pairs: np.ndarray
    evaluation_items: np.ndarray
    item_popularity: np.ndarray
    score_matrix: np.ndarray
    item_categories: np.ndarray
    user_id_map: dict[int, int]
    item_id_map: dict[int, int]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "ciao_dataset.npz",
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            centrality=self.centrality,
            train_matrix=self.train_matrix.astype(np.uint8),
            val_matrix=self.val_matrix.astype(np.uint8),
            test_matrix=self.test_matrix.astype(np.uint8),
            train_pairs=self.train_pairs,
            evaluation_items=self.evaluation_items,
            item_popularity=self.item_popularity,
            score_matrix=self.score_matrix,
            item_categories=self.item_categories,
        )
        meta = {
            "config": self.config.to_dict(),
            "num_edges": int(self.edge_index.shape[1]),
            "num_train_pairs": int(self.train_pairs.shape[0]),
            "num_users": int(self.config.num_users),
            "num_items": int(self.config.num_items),
            "num_categories": int(self.config.num_categories),
            "user_id_map_size": len(self.user_id_map),
            "item_id_map_size": len(self.item_id_map),
        }
        (output_dir / "ciao_dataset_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

    def to_torch(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "edge_index": torch.as_tensor(self.edge_index, dtype=torch.long, device=device),
            "edge_weight": torch.as_tensor(self.edge_weight, dtype=torch.float32, device=device),
            "centrality": torch.as_tensor(self.centrality, dtype=torch.float32, device=device),
        }


class CiaoDatasetLoader:
    """Load Ciao .mat files and build a ``CiaoDataset`` ready for training."""

    def __init__(self, config: CiaoDatasetConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def load(self) -> CiaoDataset:
        ratings_raw, trust_raw = self._load_mat_files()
        ratings = self._filter_positive_ratings(ratings_raw)
        users_kept, items_kept, ratings = self._k_core_filter(ratings)
        user_id_map = {int(orig): idx for idx, orig in enumerate(users_kept)}
        item_id_map = {int(orig): idx for idx, orig in enumerate(items_kept)}
        num_users = len(user_id_map)
        num_items = len(item_id_map)

        rating_users = np.array([user_id_map[int(u)] for u in ratings[:, 0]], dtype=np.int64)
        rating_items = np.array([item_id_map[int(i)] for i in ratings[:, 1]], dtype=np.int64)
        rating_categories = ratings[:, 2].astype(np.int64)

        unique_categories = np.unique(rating_categories)
        category_map = {int(orig): idx for idx, orig in enumerate(unique_categories)}
        rating_category_idx = np.array([category_map[int(c)] for c in rating_categories], dtype=np.int64)

        item_categories = self._infer_item_categories(rating_items, rating_category_idx, num_items)
        config = self.config.with_resolved_sizes(
            num_users=num_users,
            num_items=num_items,
            num_categories=len(category_map),
        )

        train_matrix, val_matrix, test_matrix = self._split_per_user(
            rating_users=rating_users,
            rating_items=rating_items,
            num_users=num_users,
            num_items=num_items,
        )

        edge_index = self._build_trust_graph(trust_raw, user_id_map)
        edge_weight = self._estimate_edge_probabilities(edge_index, train_matrix)
        centrality = _weighted_pagerank(edge_index, edge_weight, num_users)

        evaluation_items = self._select_evaluation_items(test_matrix)
        train_pairs = np.argwhere(train_matrix).astype(np.int64)
        item_popularity = train_matrix.sum(axis=0).astype(np.int64)
        score_matrix = self._build_score_matrix(
            train_matrix=train_matrix,
            edge_index=edge_index,
            edge_weight=edge_weight,
            item_categories=item_categories,
            num_users=num_users,
            num_items=num_items,
            num_categories=len(category_map),
            config=config,
        )

        return CiaoDataset(
            config=config,
            edge_index=edge_index,
            edge_weight=edge_weight,
            centrality=centrality,
            train_matrix=train_matrix,
            val_matrix=val_matrix,
            test_matrix=test_matrix,
            train_pairs=train_pairs,
            evaluation_items=evaluation_items,
            item_popularity=item_popularity,
            score_matrix=score_matrix,
            item_categories=item_categories,
            user_id_map=user_id_map,
            item_id_map=item_id_map,
        )

    def _load_mat_files(self) -> tuple[np.ndarray, np.ndarray]:
        ciao_dir = Path(self.config.ciao_dir)
        rating_path = ciao_dir / self.config.rating_file
        trust_path = ciao_dir / self.config.trust_file
        if not rating_path.exists():
            raise FileNotFoundError(f"Ciao rating file not found: {rating_path}")
        if not trust_path.exists():
            raise FileNotFoundError(f"Ciao trust file not found: {trust_path}")
        rating_blob = scipy.io.loadmat(rating_path)
        trust_blob = scipy.io.loadmat(trust_path)
        if self.config.rating_field not in rating_blob:
            raise KeyError(
                f"Rating field '{self.config.rating_field}' missing in {rating_path.name}; "
                f"available keys: {[k for k in rating_blob if not k.startswith('__')]}"
            )
        if self.config.trust_field not in trust_blob:
            raise KeyError(
                f"Trust field '{self.config.trust_field}' missing in {trust_path.name}; "
                f"available keys: {[k for k in trust_blob if not k.startswith('__')]}"
            )
        ratings = np.asarray(rating_blob[self.config.rating_field])
        trust = np.asarray(trust_blob[self.config.trust_field])
        if ratings.ndim != 2 or ratings.shape[1] < 4:
            raise ValueError(
                f"Expected rating array shape (N, >=4) but got {ratings.shape}"
            )
        if trust.ndim != 2 or trust.shape[1] != 2:
            raise ValueError(f"Expected trust array shape (E, 2) but got {trust.shape}")
        return ratings.astype(np.int64), trust.astype(np.int64)

    def _filter_positive_ratings(self, ratings: np.ndarray) -> np.ndarray:
        mask = ratings[:, 3] >= self.config.min_rating
        kept = ratings[mask]
        unique = np.unique(kept[:, [0, 1]], axis=0, return_index=True)[1]
        return kept[np.sort(unique)]

    def _k_core_filter(self, ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = ratings[:, 0]
        items = ratings[:, 1]
        active_users = np.unique(users)
        active_items = np.unique(items)
        while True:
            user_counts = np.bincount(users)
            keep_users = np.where(user_counts >= self.config.user_core)[0]
            user_mask = np.isin(users, keep_users)
            users = users[user_mask]
            items = items[user_mask]
            ratings = ratings[user_mask]

            item_counts = np.bincount(items)
            keep_items = np.where(item_counts >= self.config.item_core)[0]
            item_mask = np.isin(items, keep_items)
            users = users[item_mask]
            items = items[item_mask]
            ratings = ratings[item_mask]

            new_users = np.unique(users)
            new_items = np.unique(items)
            if (
                new_users.shape[0] == active_users.shape[0]
                and new_items.shape[0] == active_items.shape[0]
            ):
                break
            active_users = new_users
            active_items = new_items
        if active_users.size == 0 or active_items.size == 0:
            raise RuntimeError(
                "K-core filter removed every user or item. "
                "Reduce ``user_core``/``item_core`` and try again."
            )
        return active_users, active_items, ratings

    def _infer_item_categories(
        self,
        rating_items: np.ndarray,
        rating_categories: np.ndarray,
        num_items: int,
    ) -> np.ndarray:
        item_categories = np.full(num_items, -1, dtype=np.int64)
        for item_idx in np.unique(rating_items):
            mask = rating_items == item_idx
            cats = rating_categories[mask]
            values, counts = np.unique(cats, return_counts=True)
            item_categories[item_idx] = int(values[np.argmax(counts)])
        return item_categories

    def _split_per_user(
        self,
        rating_users: np.ndarray,
        rating_items: np.ndarray,
        num_users: int,
        num_items: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train = np.zeros((num_users, num_items), dtype=bool)
        val = np.zeros((num_users, num_items), dtype=bool)
        test = np.zeros((num_users, num_items), dtype=bool)
        order = np.argsort(rating_users, kind="stable")
        sorted_users = rating_users[order]
        sorted_items = rating_items[order]
        boundaries = np.flatnonzero(np.diff(sorted_users, prepend=-1, append=sorted_users.size))
        for user_idx in range(num_users):
            mask = sorted_users == user_idx
            user_items = sorted_items[mask]
            unique_items = np.unique(user_items)
            if unique_items.size == 0:
                continue
            shuffled = self.rng.permutation(unique_items)
            if shuffled.size <= 2:
                train[user_idx, shuffled] = True
                continue
            test_count = max(1, int(round(shuffled.size * self.config.test_ratio)))
            val_count = max(1, int(round(shuffled.size * self.config.val_ratio)))
            if test_count + val_count >= shuffled.size:
                val_count = max(0, shuffled.size - test_count - self.config.min_train_per_user)
                if val_count == 0 and test_count + 1 >= shuffled.size:
                    test_count = max(1, shuffled.size - self.config.min_train_per_user)
            test_items = shuffled[:test_count]
            val_items = shuffled[test_count : test_count + val_count]
            train_items = shuffled[test_count + val_count :]
            if train_items.size == 0:
                train_items = shuffled[-self.config.min_train_per_user :]
                test_items = shuffled[: -self.config.min_train_per_user]
                val_items = np.array([], dtype=np.int64)
            train[user_idx, train_items] = True
            val[user_idx, val_items] = True
            test[user_idx, test_items] = True
        del boundaries
        return train, val, test

    def _build_trust_graph(
        self,
        trust_raw: np.ndarray,
        user_id_map: dict[int, int],
    ) -> np.ndarray:
        edges: set[tuple[int, int]] = set()
        for src, dst in trust_raw:
            src_remap = user_id_map.get(int(src))
            dst_remap = user_id_map.get(int(dst))
            if src_remap is None or dst_remap is None or src_remap == dst_remap:
                continue
            edges.add((src_remap, dst_remap))
            edges.add((dst_remap, src_remap))
        if not edges:
            raise RuntimeError(
                "Trust graph is empty after k-core filter; consider lowering "
                "``user_core`` or ``item_core``."
            )
        return np.asarray(sorted(edges), dtype=np.int64).T

    def _estimate_edge_probabilities(
        self,
        edge_index: np.ndarray,
        train_matrix: np.ndarray,
    ) -> np.ndarray:
        src, dst = edge_index
        train_csr = sp.csr_matrix(train_matrix.astype(np.uint8))
        train_csr_t = train_csr.T.tocsr()
        adoption_count = np.asarray(train_csr.sum(axis=1)).ravel().astype(np.float32)
        weights = np.empty(src.shape[0], dtype=np.float32)
        smoothing = self.config.edge_prob_smoothing
        chunk = 65_536
        for start in range(0, src.shape[0], chunk):
            stop = min(src.shape[0], start + chunk)
            chunk_src = src[start:stop]
            chunk_dst = dst[start:stop]
            src_rows = train_csr[chunk_src]
            dst_rows = train_csr[chunk_dst]
            intersections = np.asarray(src_rows.multiply(dst_rows).sum(axis=1)).ravel().astype(np.float32)
            denominator = adoption_count[chunk_src] + smoothing
            chunk_weights = (intersections + smoothing * self.config.edge_prob_floor) / np.clip(denominator, 1e-3, None)
            weights[start:stop] = chunk_weights
        del train_csr_t
        weights = np.clip(weights, self.config.edge_prob_floor, self.config.edge_prob_ceiling)
        return weights.astype(np.float32)

    def _select_evaluation_items(self, test_matrix: np.ndarray) -> np.ndarray:
        popularity = test_matrix.sum(axis=0)
        candidates = np.flatnonzero(popularity > 0)
        if candidates.size == 0:
            return np.arange(min(self.config.num_eval_items, test_matrix.shape[1]), dtype=np.int64)
        ordered = candidates[np.argsort(popularity[candidates])[::-1]]
        return ordered[: self.config.num_eval_items].astype(np.int64)

    def _build_score_matrix(
        self,
        train_matrix: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
        item_categories: np.ndarray,
        num_users: int,
        num_items: int,
        num_categories: int,
        config: CiaoDatasetConfig,
    ) -> np.ndarray:
        item_popularity = train_matrix.sum(axis=0).astype(np.float32)
        log_pop = np.log1p(item_popularity)
        log_pop_z = (log_pop - log_pop.mean()) / (log_pop.std() + 1e-6)

        train_csr = sp.csr_matrix(train_matrix.astype(np.float32))
        adjacency = sp.csr_matrix(
            (edge_weight.astype(np.float32), (edge_index[0], edge_index[1])),
            shape=(num_users, num_users),
        )
        row_sums = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
        normalizer = sp.diags(1.0 / np.clip(row_sums, 1e-3, None))
        neighbor_train_rate = (normalizer @ adjacency @ train_csr).toarray().astype(np.float32)

        category_user_pref = np.zeros((num_users, max(num_categories, 1)), dtype=np.float32)
        for user_idx in range(num_users):
            user_items = np.flatnonzero(train_matrix[user_idx])
            if user_items.size == 0:
                continue
            user_cats = item_categories[user_items]
            user_cats = user_cats[user_cats >= 0]
            if user_cats.size == 0:
                continue
            counts = np.bincount(user_cats, minlength=num_categories).astype(np.float32)
            category_user_pref[user_idx] = counts / counts.sum()
        item_category_indicator = np.zeros((num_items, max(num_categories, 1)), dtype=np.float32)
        for item_idx in range(num_items):
            cat = int(item_categories[item_idx])
            if cat >= 0:
                item_category_indicator[item_idx, cat] = 1.0
        category_match = category_user_pref @ item_category_indicator.T
        category_match_z = (category_match - category_match.mean()) / (category_match.std() + 1e-6)

        logits = (
            config.score_bias
            + config.score_pop_weight * log_pop_z[None, :]
            + config.score_neighbor_weight * neighbor_train_rate
            + config.score_category_weight * category_match_z
        )
        score_matrix = _sigmoid(logits).astype(np.float32)
        score_matrix = np.clip(score_matrix, 1e-4, 1.0 - 1e-4)
        return score_matrix
