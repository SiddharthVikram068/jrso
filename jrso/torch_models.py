"""Torch models for the JRSO benchmark.

This module implements the recommendation, influence-only and joint
recommendation+seeding models that the torch / ciao experiments train
and evaluate.

Design notes
============

The encoder is a LightGCN-style bipartite + social GNN. It propagates
user/item embeddings on three jointly-normalised adjacencies:

* user-item (bipartite) interactions from the training matrix
* user-user social trust edges
* item-item co-interaction edges (only used by JRSO_third)

LightGCN drops feature transforms and non-linearities and just averages
embeddings across layers; on Ciao that is a much stronger backbone than
the older social-only encoder this repository used to ship with.

Joint models follow the JRSO formulation in the report:

* continuous seed vector s = 1 - exp(-K * softmax(seed_logits))
* differentiable Independent Cascade (Message-Passing Diffusion) with
  edge influence probabilities P_{v,u}(Theta) parameterised by the
  recommender's GNN embeddings
* joint loss L_joint = L_rec - lambda * sum_u h^{(T)}_u(Theta, s)

At evaluation time we pick seeds via a greedy CELF-style coverage
selector that uses the *learned* edge probabilities to pick a small
high-affinity, low-redundancy subset of users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from .torch_config import TorchTrainingConfig


def _build_lightgcn_adjacency(
    train_matrix: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    num_users: int,
    num_items: int,
    social_scale: float = 0.5,
) -> sp.csr_matrix:
    """Build symmetrically normalised LightGCN adjacency.

    The combined graph has ``num_users + num_items`` nodes:

    * bipartite (user-item) edges: weight 1.0
    * social (user-user) edges: weight ``social_scale * edge_weight``

    The output is the symmetrically normalised matrix
    ``D^{-1/2} A D^{-1/2}`` returned as ``scipy.sparse.csr_matrix``.
    """

    train_csr = sp.csr_matrix(train_matrix.astype(np.float32))
    user_to_item = sp.hstack(
        [
            sp.csr_matrix((num_users, num_users), dtype=np.float32),
            train_csr,
        ],
        format="csr",
    )
    item_to_user = sp.hstack(
        [
            train_csr.T.tocsr(),
            sp.csr_matrix((num_items, num_items), dtype=np.float32),
        ],
        format="csr",
    )
    bipartite = sp.vstack([user_to_item, item_to_user], format="csr")

    if edge_index.size > 0 and edge_weight.size > 0:
        weights = social_scale * edge_weight.astype(np.float32)
        social_block = sp.csr_matrix(
            (weights, (edge_index[0], edge_index[1])),
            shape=(num_users, num_users),
            dtype=np.float32,
        )
        zero_users_items = sp.csr_matrix((num_users, num_items), dtype=np.float32)
        social_full_top = sp.hstack([social_block, zero_users_items], format="csr")
        zero_items_full = sp.csr_matrix((num_items, num_users + num_items), dtype=np.float32)
        social_full = sp.vstack([social_full_top, zero_items_full], format="csr")
        adjacency = bipartite + social_full
    else:
        adjacency = bipartite

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    diag = sp.diags(inv_sqrt)
    return (diag @ adjacency @ diag).tocsr()


def _csr_to_torch_sparse(matrix: sp.csr_matrix) -> torch.sparse.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).coalesce()


def continuous_seed_probabilities(logits: torch.Tensor, budget: int) -> torch.Tensor:
    """Continuous relaxation of "pick K seeds": ``s_u = 1 - exp(-K * softmax(logits)_u)``."""

    weights = torch.softmax(logits, dim=0)
    return 1.0 - torch.exp(-float(budget) * weights)


@dataclass
class TorchModelArtifacts:
    history: list[dict[str, float]]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class LightGCNEncoder(nn.Module):
    """LightGCN-style bipartite encoder with optional social edges.

    Outputs separate ``(num_users, hidden_dim)`` and
    ``(num_items, hidden_dim)`` representations after averaging the
    embeddings produced at every propagation step.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, num_layers)
        self.dropout = dropout
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def all_embeddings(self) -> torch.Tensor:
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

    def forward(self, lightgcn_adj: torch.sparse.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.all_embeddings()
        layer_outputs = [embeddings]
        current = embeddings
        for _ in range(self.num_layers):
            current = torch.sparse.mm(lightgcn_adj, current)
            if self.dropout > 0.0 and self.training:
                current = F.dropout(current, p=self.dropout)
            layer_outputs.append(current)
        stacked = torch.stack(layer_outputs, dim=0)
        averaged = stacked.mean(dim=0)
        user_repr = averaged[: self.num_users]
        item_repr = averaged[self.num_users :]
        return user_repr, item_repr


# ---------------------------------------------------------------------------
# Base recommender
# ---------------------------------------------------------------------------


class BaseTorchRecommender(nn.Module):
    """Common recommender backbone used by every torch model variant."""

    model_name = "Base"

    def __init__(
        self,
        dataset: Any,
        config: TorchTrainingConfig,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.num_users = int(dataset.config.num_users)
        self.num_items = int(dataset.config.num_items)
        self.encoder = LightGCNEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.artifacts = TorchModelArtifacts(history=[], metadata={})

    # ---------------- representations ----------------
    def encode(
        self,
        graph: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(graph["lightgcn_adj"])

    def encode_users(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        user_repr, _ = self.encode(graph)
        return user_repr

    def item_representations(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        _, item_repr = self.encode(graph)
        return item_repr

    # ---------------- scoring ----------------
    def score_all(
        self,
        graph: dict[str, torch.Tensor],
        user_repr: torch.Tensor | None = None,
        item_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if user_repr is None or item_repr is None:
            user_repr, item_repr = self.encode(graph)
        scaled = (user_repr @ item_repr.T) / float(np.sqrt(self.config.hidden_dim))
        scaled = scaled + self.user_bias.weight + self.item_bias.weight.T + self.global_bias
        return scaled

    def predict_score_tensor(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.score_all(graph)

    def predict_score_matrix(self, graph: dict[str, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            scores = self.score_all(graph)
        return scores.detach().cpu().numpy()

    # ---------------- losses ----------------
    def _bpr_loss(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        scale = float(np.sqrt(self.config.hidden_dim))
        u = user_repr[users]
        pos = item_repr[pos_items]
        neg = item_repr[neg_candidates]

        pos_score = (u * pos).sum(dim=-1) / scale + self.user_bias(users).squeeze(-1) + self.item_bias(pos_items).squeeze(-1) + self.global_bias
        neg_score = (u.unsqueeze(1) * neg).sum(dim=-1) / scale + self.user_bias(users) + self.item_bias(neg_candidates).squeeze(-1) + self.global_bias

        if self.config.use_hard_negatives:
            hardest = neg_score.max(dim=1).values
            base_loss = F.softplus(hardest - pos_score).mean()
            mean_loss = F.softplus(neg_score.mean(dim=1) - pos_score).mean()
            loss = 0.7 * base_loss + 0.3 * mean_loss
        else:
            loss = F.softplus(neg_score.mean(dim=1) - pos_score).mean()

        # In-batch negatives: also push positive scores above scores against
        # other users' positives in the same batch.
        in_batch_pos = item_repr[pos_items]
        in_batch_scores = (u @ in_batch_pos.T) / scale + self.user_bias(users) + self.item_bias(pos_items).squeeze(-1).unsqueeze(0) + self.global_bias
        in_batch_labels = torch.arange(users.shape[0], device=users.device)
        in_batch_loss = F.cross_entropy(in_batch_scores, in_batch_labels)
        loss = loss + 0.5 * in_batch_loss

        # Regulariser on raw (un-propagated) embeddings, LightGCN-style.
        reg = (
            self.encoder.user_embedding(users).pow(2).sum()
            + self.encoder.item_embedding(pos_items).pow(2).sum()
            + self.encoder.item_embedding(neg_candidates).pow(2).sum()
        ) / float(users.shape[0])
        loss = loss + 1e-4 * reg

        metrics = {
            "rec_loss": float(loss.detach().item()),
            "pos_score": float(pos_score.detach().mean().item()),
            "neg_score": float(neg_score.detach().mean().item()),
        }
        return loss, metrics

    def compute_loss(
        self,
        graph: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del epoch
        user_repr, item_repr = self.encode(graph)
        loss, metrics = self._bpr_loss(
            user_repr=user_repr,
            item_repr=item_repr,
            users=batch["users"],
            pos_items=batch["pos_items"],
            neg_candidates=batch["neg_candidates"],
        )
        metrics["spread_bonus"] = 0.0
        return loss, metrics

    # ---------------- seeding ----------------
    def _affinity_for_item(
        self,
        user_repr: torch.Tensor,
        item_vector: torch.Tensor,
    ) -> torch.Tensor:
        scale = float(np.sqrt(self.config.hidden_dim))
        return (user_repr * item_vector.unsqueeze(0)).sum(dim=-1) / scale + self.user_bias.weight.squeeze(-1)

    def select_seeds(
        self,
        graph: dict[str, torch.Tensor],
        item_idx: int,
        user_repr: torch.Tensor | None = None,
        item_repr: torch.Tensor | None = None,
    ) -> list[int]:
        with torch.no_grad():
            if user_repr is None or item_repr is None:
                user_repr, item_repr = self.encode(graph)
            affinity = self._affinity_for_item(user_repr, item_repr[item_idx])
            _, indices = torch.topk(affinity, k=self.dataset.config.k_seeds)
        return [int(value) for value in indices.detach().cpu().tolist()]


class TorchRecSysOnly(BaseTorchRecommender):
    model_name = "RecSys-Only"


# ---------------------------------------------------------------------------
# Joint models
# ---------------------------------------------------------------------------


def _greedy_coverage_seeds(
    affinity: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_prob: np.ndarray,
    centrality: np.ndarray,
    k: int,
    candidate_pool_size: int = 256,
    diversity_weight: float = 1.0,
    centrality_weight: float = 0.10,
) -> list[int]:
    """Vectorised CELF-style greedy seed selection.

    Picks ``k`` users that jointly maximise expected one-hop weighted
    coverage of high-affinity nodes, while penalising overlap among
    already-chosen seeds. Each round computes the marginal gain for
    every candidate in a single vectorised pass via ``np.add.at`` so
    cost per round is ``O(num_pool_edges)`` not ``O(pool_size *
    num_pool_edges)``.
    """

    num_users = affinity.shape[0]
    if num_users == 0:
        return []
    pool_size = min(num_users, max(k * 6, candidate_pool_size))
    centrality_norm = centrality / max(float(centrality.max()), 1e-8)
    base_score = affinity + centrality_weight * centrality_norm
    if pool_size < num_users:
        candidates = np.argpartition(-base_score, kth=pool_size - 1)[:pool_size]
    else:
        candidates = np.arange(num_users, dtype=np.int64)
    candidate_to_local = -np.ones(num_users, dtype=np.int64)
    candidate_to_local[candidates] = np.arange(candidates.shape[0], dtype=np.int64)

    src_local = candidate_to_local[edge_src]
    in_pool = src_local >= 0
    pool_local_src = src_local[in_pool]
    pool_dst = edge_dst[in_pool]
    pool_prob = edge_prob[in_pool].astype(np.float32)

    residual = np.ones(num_users, dtype=np.float32)
    selected = np.zeros(pool_size, dtype=bool)
    seeds: list[int] = []
    affinity_pool = affinity[candidates]
    centrality_pool = centrality_norm[candidates]

    while len(seeds) < min(k, pool_size):
        gains = affinity_pool * residual[candidates] + centrality_weight * centrality_pool
        if pool_local_src.size > 0:
            contributions = pool_prob * residual[pool_dst] * affinity[pool_dst]
            agg = np.zeros(pool_size, dtype=np.float32)
            np.add.at(agg, pool_local_src, contributions)
            gains = gains + diversity_weight * agg
        gains[selected] = -np.inf
        best_local = int(np.argmax(gains))
        if not np.isfinite(gains[best_local]):
            break
        chosen = int(candidates[best_local])
        seeds.append(chosen)
        selected[best_local] = True
        residual[chosen] = 0.0
        chosen_mask = pool_local_src == best_local
        if chosen_mask.any():
            neigh_dst = pool_dst[chosen_mask]
            neigh_prob = pool_prob[chosen_mask]
            residual[neigh_dst] = residual[neigh_dst] * (1.0 - neigh_prob)
    return seeds


class TorchJRSO(BaseTorchRecommender):
    """End-to-end joint recommender + seeding model from the report.

    * Affinity = LightGCN inner product score (the recommender output).
    * Seed logits = MLP over [user_repr, item_repr, user_repr*item_repr,
      |user_repr - item_repr|, centrality, propagation hint] + affinity.
    * Edge influence probabilities P_{v,u}(Theta) = trust_weight *
      sigmoid(GNN-derived edge logit) * sigmoid(affinity_v) *
      sigmoid(affinity_u).
    * Continuous seeding via softmax * K.
    * Differentiable Independent Cascade with the formula from the
      report ``h_u^{t} = 1 - (1 - h_u^{t-1}) * prod_v (1 - P_{v,u} * h_v^{t-1})``.
    * Joint loss: ``L_rec - lambda * mean_u h_u^T``.
    """

    model_name = "JRSO"

    overlap_weight = 0.05
    edge_alignment_weight = 0.05

    def __init__(self, dataset: Any, config: TorchTrainingConfig) -> None:
        super().__init__(dataset, config)
        hidden_dim = config.hidden_dim
        self.seed_bias = nn.Parameter(torch.zeros(self.num_users))

        self.seed_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.src_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dst_projection = nn.Linear(hidden_dim, hidden_dim)
        self.item_projection = nn.Linear(hidden_dim, hidden_dim)
        self.edge_temperature = nn.Parameter(torch.tensor(1.0))

    # ----- helpers -----
    def _propagation_hint(
        self,
        affinity: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        src, dst = graph["edge_index"]
        edge_weight = graph["edge_weight"].to(dtype=affinity.dtype)
        propagation = torch.zeros_like(affinity)
        propagation.index_add_(0, src, edge_weight * affinity[dst])
        normaliser = propagation.max().clamp(min=1e-6)
        return propagation / normaliser

    def _select_diffusion_edges(
        self,
        graph: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """During training, subsample edges to speed up the diffusion bonus.

        Returns ``(src, dst, weight)`` tensors. At eval time we use the full
        graph; at train time we cap to roughly ``training_edge_cap`` edges
        and rescale the weights so the expected per-node propagation
        magnitude stays comparable.
        """

        src, dst = graph["edge_index"]
        edge_weight = graph["edge_weight"].to(dtype=torch.float32)
        if not self.training:
            return src, dst, edge_weight
        cap = int(getattr(self.config, "diffusion_edge_cap", 32_768))
        num_edges = src.shape[0]
        if cap <= 0 or cap >= num_edges:
            return src, dst, edge_weight
        sampled = torch.randperm(num_edges, device=src.device)[:cap]
        scale = float(num_edges) / float(cap)
        return src[sampled], dst[sampled], (edge_weight[sampled] * scale).clamp(max=1.0)

    def _seed_logits(
        self,
        user_repr: torch.Tensor,
        item_vector: torch.Tensor,
        graph: dict[str, torch.Tensor],
        affinity: torch.Tensor,
    ) -> torch.Tensor:
        item_expanded = item_vector.unsqueeze(0).expand_as(user_repr)
        centrality = graph["centrality"].to(dtype=user_repr.dtype).unsqueeze(1)
        propagation = self._propagation_hint(affinity, graph).unsqueeze(1)
        features = torch.cat(
            [
                user_repr,
                item_expanded,
                user_repr * item_expanded,
                torch.abs(user_repr - item_expanded),
                centrality,
                propagation,
            ],
            dim=-1,
        )
        return self.seed_head(features).squeeze(-1) + 0.5 * affinity + self.seed_bias

    def _edge_probabilities(
        self,
        user_repr: torch.Tensor,
        item_vector: torch.Tensor,
        graph: dict[str, torch.Tensor],
        affinity_sigmoid: torch.Tensor,
        edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if edges is None:
            src, dst = graph["edge_index"]
            edge_weight = graph["edge_weight"].to(dtype=user_repr.dtype)
        else:
            src, dst, edge_weight = edges
            edge_weight = edge_weight.to(dtype=user_repr.dtype)
        scale = float(np.sqrt(self.config.hidden_dim))
        src_repr = self.src_projection(user_repr[src])
        dst_repr = self.dst_projection(user_repr[dst])
        item_context = self.item_projection(item_vector)
        edge_logits = (
            (src_repr * dst_repr).sum(dim=-1)
            + (src_repr * item_context.unsqueeze(0)).sum(dim=-1)
            + (dst_repr * item_context.unsqueeze(0)).sum(dim=-1)
        ) / scale
        edge_logits = edge_logits / F.softplus(self.edge_temperature).clamp(min=1e-2)
        edge_prob = (
            edge_weight
            * torch.sigmoid(edge_logits)
            * affinity_sigmoid[src]
            * affinity_sigmoid[dst]
        )
        return edge_prob.clamp(max=0.95)

    def _diffusion(
        self,
        user_repr: torch.Tensor,
        item_vector: torch.Tensor,
        graph: dict[str, torch.Tensor],
        edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        affinity = self._affinity_for_item(user_repr, item_vector)
        affinity_sigmoid = torch.sigmoid(affinity)
        seed_logits = self._seed_logits(user_repr, item_vector, graph, affinity)
        seed_prob = continuous_seed_probabilities(seed_logits, self.dataset.config.k_seeds)
        edge_prob = self._edge_probabilities(user_repr, item_vector, graph, affinity_sigmoid, edges=edges)
        if edges is None:
            src, dst = graph["edge_index"]
        else:
            src, dst = edges[0], edges[1]
        hidden = seed_prob
        for _ in range(self.config.diffusion_steps):
            safe_term = torch.clamp(1.0 - edge_prob * hidden[src], min=1e-6, max=1.0)
            log_prod = torch.zeros(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
            log_prod.index_add_(0, dst, safe_term.log())
            hidden = 1.0 - (1.0 - hidden) * log_prod.exp()
        return hidden, seed_prob, edge_prob

    def _spread_objective(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        graph: dict[str, torch.Tensor],
        item_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spreads: list[torch.Tensor] = []
        overlaps: list[torch.Tensor] = []
        edges = self._select_diffusion_edges(graph) if self.training else None
        if edges is None:
            src, dst = graph["edge_index"]
        else:
            src, dst = edges[0], edges[1]
        for item_idx in item_indices:
            hidden, seed_prob, edge_prob = self._diffusion(
                user_repr=user_repr,
                item_vector=item_repr[item_idx],
                graph=graph,
                edges=edges,
            )
            affinity = torch.sigmoid(self._affinity_for_item(user_repr, item_repr[item_idx]))
            spreads.append((hidden * affinity).mean())
            overlaps.append((seed_prob[src] * seed_prob[dst] * edge_prob).mean())
        spread_bonus = (
            torch.stack(spreads).mean() if spreads else torch.tensor(0.0, device=user_repr.device)
        )
        overlap_penalty = (
            torch.stack(overlaps).mean() if overlaps else torch.tensor(0.0, device=user_repr.device)
        )
        return spread_bonus, overlap_penalty

    def _edge_alignment_loss(
        self,
        user_repr: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        src, dst = graph["edge_index"]
        if src.shape[0] == 0:
            return torch.tensor(0.0, device=user_repr.device)
        sample_size = min(src.shape[0], 16_384)
        sampled = torch.randperm(src.shape[0], device=src.device)[:sample_size]
        scale = float(np.sqrt(self.config.hidden_dim))
        edge_src = src[sampled]
        edge_dst = dst[sampled]
        logits = (
            self.src_projection(user_repr[edge_src]) * self.dst_projection(user_repr[edge_dst])
        ).sum(dim=-1) / scale
        return F.mse_loss(torch.sigmoid(logits), graph["edge_weight"][sampled])

    # ----- training loss -----
    def compute_loss(
        self,
        graph: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        user_repr, item_repr = self.encode(graph)
        rec_loss, metrics = self._bpr_loss(
            user_repr=user_repr,
            item_repr=item_repr,
            users=batch["users"],
            pos_items=batch["pos_items"],
            neg_candidates=batch["neg_candidates"],
        )

        unique_items = torch.unique(batch["pos_items"])
        if unique_items.numel() > self.config.spread_items_per_step:
            permutation = torch.randperm(unique_items.numel(), device=unique_items.device)
            unique_items = unique_items[permutation[: self.config.spread_items_per_step]]

        spread_bonus, overlap_penalty = self._spread_objective(
            user_repr=user_repr,
            item_repr=item_repr,
            graph=graph,
            item_indices=unique_items,
        )
        edge_alignment = self._edge_alignment_loss(user_repr, graph)
        warmup = max(1, self.config.warmup_epochs)
        lambda_scale = min(1.0, float(epoch + 1) / float(warmup))
        loss = (
            rec_loss
            - (self.config.lambda_spread * lambda_scale * spread_bonus)
            + self.overlap_weight * overlap_penalty
            + self.edge_alignment_weight * edge_alignment
        )
        metrics["spread_bonus"] = float(spread_bonus.detach().item())
        metrics["edge_alignment"] = float(edge_alignment.detach().item())
        return loss, metrics

    # ----- seed selection -----
    def select_seeds(
        self,
        graph: dict[str, torch.Tensor],
        item_idx: int,
        user_repr: torch.Tensor | None = None,
        item_repr: torch.Tensor | None = None,
    ) -> list[int]:
        with torch.no_grad():
            if user_repr is None or item_repr is None:
                user_repr, item_repr = self.encode(graph)
            item_vector = item_repr[item_idx]
            affinity = self._affinity_for_item(user_repr, item_vector)
            affinity_sigmoid = torch.sigmoid(affinity)
            edge_prob = self._edge_probabilities(user_repr, item_vector, graph, affinity_sigmoid)

        affinity_np = affinity_sigmoid.detach().cpu().numpy().astype(np.float32)
        edge_src = graph["edge_index"][0].detach().cpu().numpy()
        edge_dst = graph["edge_index"][1].detach().cpu().numpy()
        edge_prob_np = edge_prob.detach().cpu().numpy().astype(np.float32)
        centrality_np = graph["centrality"].detach().cpu().numpy().astype(np.float32)
        return _greedy_coverage_seeds(
            affinity=affinity_np,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_prob=edge_prob_np,
            centrality=centrality_np,
            k=self.dataset.config.k_seeds,
            candidate_pool_size=max(256, 6 * self.dataset.config.k_seeds),
            diversity_weight=1.0,
            centrality_weight=0.10,
        )


class TorchJRSOSecond(TorchJRSO):
    """JRSO variant with attention-style edge probabilities and stronger overlap regularisation."""

    model_name = "JRSO_second"
    overlap_weight = 0.10
    edge_alignment_weight = 0.08

    def __init__(self, dataset: Any, config: TorchTrainingConfig) -> None:
        super().__init__(dataset, config)
        hidden_dim = config.hidden_dim
        self.attention_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _edge_probabilities(
        self,
        user_repr: torch.Tensor,
        item_vector: torch.Tensor,
        graph: dict[str, torch.Tensor],
        affinity_sigmoid: torch.Tensor,
        edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if edges is None:
            src, dst = graph["edge_index"]
            edge_weight = graph["edge_weight"].to(dtype=user_repr.dtype)
        else:
            src, dst, edge_weight = edges
            edge_weight = edge_weight.to(dtype=user_repr.dtype)
        item_expanded = item_vector.unsqueeze(0).expand(src.shape[0], -1)
        features = torch.cat(
            [
                self.src_projection(user_repr[src]),
                self.dst_projection(user_repr[dst]),
                self.item_projection(item_expanded),
            ],
            dim=-1,
        )
        attention_logits = self.attention_head(features).squeeze(-1)
        attention_logits = attention_logits / F.softplus(self.edge_temperature).clamp(min=1e-2)
        edge_prob = (
            edge_weight
            * torch.sigmoid(attention_logits)
            * affinity_sigmoid[src]
            * affinity_sigmoid[dst]
        )
        return edge_prob.clamp(max=0.95)


class TorchJRSOThird(TorchJRSO):
    """JRSO variant inspired by DiffNet++.

    Adds a learned gated fusion between the LightGCN-propagated user
    representation and an interaction-history representation built from
    the user's training-set items. Tends to be the strongest joint
    model on Ciao because the item-history signal complements the
    relatively sparse trust graph.
    """

    model_name = "JRSO_third"
    overlap_weight = 0.06
    edge_alignment_weight = 0.06

    def __init__(self, dataset: Any, config: TorchTrainingConfig) -> None:
        super().__init__(dataset, config)
        hidden_dim = config.hidden_dim
        interaction_weights = dataset.train_matrix.astype(np.float32)
        row_sum = np.clip(interaction_weights.sum(axis=1, keepdims=True), 1.0, None)
        interaction_weights /= row_sum
        self.register_buffer(
            "interaction_weights",
            torch.as_tensor(interaction_weights, dtype=torch.float32),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode(self, graph: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        user_repr, item_repr = super().encode(graph)
        history_repr = self.interaction_weights.to(item_repr.device) @ item_repr
        centrality = graph["centrality"].to(dtype=user_repr.dtype).unsqueeze(1)
        gate_features = torch.cat([user_repr, history_repr, centrality], dim=-1)
        gate = torch.sigmoid(self.gate_head(gate_features))
        residual = self.history_projection(
            torch.cat([user_repr * history_repr, torch.abs(user_repr - history_repr), centrality], dim=-1)
        )
        fused_user = gate * user_repr + (1.0 - gate) * history_repr + 0.20 * residual
        return fused_user, item_repr


# ---------------------------------------------------------------------------
# Non-trainable baselines
# ---------------------------------------------------------------------------


class InfluenceOnlyBaseline:
    """Centrality-greedy IM baseline: pick high-centrality users with neighbour suppression."""

    model_name = "IM-Only"

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self.global_seeds = self._build_global_seeds()
        self.artifacts = TorchModelArtifacts(history=[], metadata={"global_seeds": self.global_seeds})

    def _build_global_seeds(self) -> list[int]:
        scores = self.dataset.centrality.copy()
        src, dst = self.dataset.edge_index
        edge_weight = self.dataset.edge_weight.astype(np.float32)
        residual = np.ones(scores.shape[0], dtype=np.float32)
        order: list[int] = []
        for _ in range(self.dataset.config.k_seeds):
            base = scores * residual
            best_idx = int(np.argmax(base))
            order.append(best_idx)
            residual[best_idx] = 0.0
            mask = src == best_idx
            if mask.any():
                neighbours = dst[mask]
                weights = edge_weight[mask]
                residual[neighbours] = residual[neighbours] * (1.0 - 0.6 * weights)
        return order

    def select_seeds(self, item_idx: int) -> list[int]:
        del item_idx
        return list(self.global_seeds)


class TwoStageBaseline:
    """First train RecSys-Only, then pick seeds via affinity * (1 + influence pre-aggregate)."""

    model_name = "Two-Stage"

    def __init__(
        self,
        dataset: Any,
        recommender: BaseTorchRecommender,
        graph: dict[str, torch.Tensor],
    ) -> None:
        self.dataset = dataset
        self.recommender = recommender
        self.graph = graph
        self.artifacts = TorchModelArtifacts(
            history=list(recommender.artifacts.history),
            metadata={"derived_from": recommender.model_name},
        )

    def predict_score_matrix(self, graph: dict[str, torch.Tensor]) -> np.ndarray:
        return self.recommender.predict_score_matrix(graph)

    def select_seeds(
        self,
        item_idx: int,
        user_repr: torch.Tensor | None = None,
        item_repr: torch.Tensor | None = None,
    ) -> list[int]:
        with torch.no_grad():
            if user_repr is None or item_repr is None:
                user_repr, item_repr = self.recommender.encode(self.graph)
            affinity = torch.sigmoid(
                self.recommender._affinity_for_item(user_repr, item_repr[item_idx])
            )
            src, dst = self.graph["edge_index"]
            edge_weight = self.graph["edge_weight"]
            propagation = torch.zeros(self.dataset.config.num_users, device=affinity.device)
            propagation.index_add_(0, src, edge_weight * affinity[dst])
            propagation = propagation / propagation.max().clamp(min=1e-6)
            centrality = self.graph["centrality"].to(dtype=affinity.dtype)
            seed_scores = affinity * (1.0 + 0.7 * propagation + 0.30 * centrality)
        affinity_np = affinity.detach().cpu().numpy().astype(np.float32)
        edge_src = src.detach().cpu().numpy()
        edge_dst = dst.detach().cpu().numpy()
        edge_prob_np = (
            edge_weight.detach().cpu().numpy().astype(np.float32)
            * affinity_np[edge_src]
            * affinity_np[edge_dst]
        )
        centrality_np = centrality.detach().cpu().numpy().astype(np.float32)
        return _greedy_coverage_seeds(
            affinity=affinity_np,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_prob=np.clip(edge_prob_np, 0.0, 0.95),
            centrality=centrality_np,
            k=self.dataset.config.k_seeds,
            candidate_pool_size=max(256, 6 * self.dataset.config.k_seeds),
            diversity_weight=0.7,
            centrality_weight=0.10,
        )
