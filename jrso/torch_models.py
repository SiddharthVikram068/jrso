from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .torch_config import TorchTrainingConfig
from .torch_dataset import SparseSyntheticDataset


def continuous_seed_probabilities(logits: torch.Tensor, budget: int) -> torch.Tensor:
    weights = torch.softmax(logits, dim=0)
    return 1.0 - torch.exp(-budget * weights)


def _score_matrix(user_repr: torch.Tensor, item_repr: torch.Tensor) -> torch.Tensor:
    return (user_repr @ item_repr.T) / np.sqrt(user_repr.shape[1])


@dataclass
class TorchModelArtifacts:
    history: list[dict[str, float]]
    metadata: dict[str, Any]


class SparseResidualLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, x[src])
        degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        aggregated = aggregated / degree.clamp(min=1.0).unsqueeze(1)
        update = self.self_linear(x) + self.neigh_linear(aggregated)
        update = F.gelu(update)
        update = self.dropout(update)
        return self.norm(x + update)


class SocialEncoder(nn.Module):
    def __init__(self, num_users: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.layers = nn.ModuleList(
            [SparseResidualLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        nn.init.xavier_uniform_(self.user_embedding.weight)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = self.user_embedding.weight
        for layer in self.layers:
            hidden = layer(hidden, edge_index)
        return F.normalize(hidden, dim=-1)


class BaseTorchRecommender(nn.Module):
    model_name = "Base"

    def __init__(
        self,
        dataset: SparseSyntheticDataset,
        config: TorchTrainingConfig,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.encoder = SocialEncoder(
            num_users=dataset.config.num_users,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.item_embedding = nn.Embedding(dataset.config.num_items, config.hidden_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.artifacts = TorchModelArtifacts(history=[], metadata={})

    def encode_users(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(graph["edge_index"])

    def item_representations(self) -> torch.Tensor:
        return F.normalize(self.item_embedding.weight, dim=-1)

    def predict_score_tensor(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        user_repr = self.encode_users(graph)
        item_repr = self.item_representations()
        return _score_matrix(user_repr, item_repr)

    def predict_score_matrix(self, graph: dict[str, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            scores = self.predict_score_tensor(graph)
        return scores.detach().cpu().numpy()

    def _candidate_scores(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        users: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        user_batch = user_repr[users].unsqueeze(1)
        item_batch = item_repr[candidate_items]
        return (user_batch * item_batch).sum(dim=-1) / np.sqrt(user_repr.shape[1])

    def _recommendation_loss(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pos_scores = (user_repr[users] * item_repr[pos_items]).sum(dim=-1) / np.sqrt(user_repr.shape[1])
        candidate_scores = self._candidate_scores(user_repr, item_repr, users, neg_candidates)
        if self.config.use_hard_negatives:
            neg_scores = candidate_scores.max(dim=1).values
        else:
            neg_scores = candidate_scores[:, 0]
        loss = F.softplus(neg_scores - pos_scores).mean()
        metrics = {
            "rec_loss": float(loss.detach().item()),
            "pos_score": float(pos_scores.detach().mean().item()),
            "neg_score": float(neg_scores.detach().mean().item()),
        }
        return loss, metrics

    def compute_loss(
        self,
        graph: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del epoch
        user_repr = self.encode_users(graph)
        item_repr = self.item_representations()
        rec_loss, metrics = self._recommendation_loss(
            user_repr=user_repr,
            item_repr=item_repr,
            users=batch["users"],
            pos_items=batch["pos_items"],
            neg_candidates=batch["neg_candidates"],
        )
        metrics["spread_bonus"] = 0.0
        return rec_loss, metrics

    def seed_scores(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del graph
        return (user_repr * item_repr.unsqueeze(0)).sum(dim=-1) / np.sqrt(user_repr.shape[1])

    def select_seeds(
        self,
        graph: dict[str, torch.Tensor],
        item_idx: int,
        user_repr: torch.Tensor | None = None,
    ) -> list[int]:
        with torch.no_grad():
            encoded = user_repr if user_repr is not None else self.encode_users(graph)
            item_repr = self.item_representations()[item_idx]
            seed_scores = self.seed_scores(encoded, item_repr, graph)
            _, indices = torch.topk(seed_scores, k=self.dataset.config.k_seeds)
        return [int(value) for value in indices.detach().cpu().tolist()]


class TorchRecSysOnly(BaseTorchRecommender):
    model_name = "RecSys-Only"


class TorchJRSO(BaseTorchRecommender):
    model_name = "JRSO"

    def __init__(self, dataset: SparseSyntheticDataset, config: TorchTrainingConfig) -> None:
        super().__init__(dataset, config)
        self.seed_bias = nn.Parameter(torch.zeros(dataset.config.num_users))

    def _affinity_logits(self, user_repr: torch.Tensor, item_repr: torch.Tensor) -> torch.Tensor:
        return (user_repr * item_repr.unsqueeze(0)).sum(dim=-1) / np.sqrt(user_repr.shape[1])

    def _seed_logits(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        graph: dict[str, torch.Tensor],
        affinity_logits: torch.Tensor,
    ) -> torch.Tensor:
        del user_repr, item_repr, graph
        return affinity_logits + self.seed_bias

    def _edge_probabilities(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        graph: dict[str, torch.Tensor],
        affinity_logits: torch.Tensor,
        affinity: torch.Tensor,
    ) -> torch.Tensor:
        del item_repr, affinity_logits
        src, dst = graph["edge_index"]
        structural = torch.sigmoid((user_repr[src] * user_repr[dst]).sum(dim=-1) / np.sqrt(user_repr.shape[1]))
        edge_prob = graph["edge_weight"] * structural * affinity[src] * affinity[dst]
        return edge_prob.clamp(max=0.95)

    @staticmethod
    def _coverage_aware_seed_selection(
        candidates: np.ndarray,
        affinity: np.ndarray,
        downstream: np.ndarray,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_prob: np.ndarray,
        centrality: np.ndarray,
        k: int,
    ) -> list[int]:
        num_users = affinity.shape[0]
        candidate_map = np.full(num_users, -1, dtype=np.int64)
        candidate_map[candidates] = np.arange(candidates.shape[0], dtype=np.int64)
        local_src = candidate_map[edge_src]
        outgoing_targets: list[list[int]] = [[] for _ in range(candidates.shape[0])]
        outgoing_probs: list[list[float]] = [[] for _ in range(candidates.shape[0])]
        for source_local, target_idx, probability in zip(local_src, edge_dst, edge_prob):
            if source_local >= 0 and probability > 0.0:
                outgoing_targets[int(source_local)].append(int(target_idx))
                outgoing_probs[int(source_local)].append(float(probability))

        residual = np.ones(num_users, dtype=np.float64)
        selected = np.zeros(candidates.shape[0], dtype=bool)
        seeds: list[int] = []
        centrality = centrality / max(float(centrality.max()), 1e-8)

        while len(seeds) < min(k, candidates.shape[0]):
            best_local = -1
            best_gain = float("-inf")
            for local_idx, node_idx in enumerate(candidates):
                if selected[local_idx]:
                    continue
                gain = float(affinity[node_idx] * residual[node_idx] + 0.05 * centrality[node_idx])
                if outgoing_targets[local_idx]:
                    targets = np.asarray(outgoing_targets[local_idx], dtype=np.int64)
                    probabilities = np.asarray(outgoing_probs[local_idx], dtype=np.float64)
                    gain += float(np.sum(residual[targets] * probabilities * downstream[targets]))
                if gain > best_gain:
                    best_gain = gain
                    best_local = local_idx
            if best_local < 0:
                break
            chosen = int(candidates[best_local])
            seeds.append(chosen)
            selected[best_local] = True
            residual[chosen] = 0.0
            if outgoing_targets[best_local]:
                targets = np.asarray(outgoing_targets[best_local], dtype=np.int64)
                probabilities = np.asarray(outgoing_probs[best_local], dtype=np.float64)
                residual[targets] *= 1.0 - probabilities
        return seeds

    def _graph_aware_seeds(
        self,
        seed_logits: torch.Tensor,
        affinity: torch.Tensor,
        graph: dict[str, torch.Tensor],
        edge_prob: torch.Tensor,
    ) -> list[int]:
        edge_src = graph["edge_index"][0].detach().cpu().numpy()
        edge_dst = graph["edge_index"][1].detach().cpu().numpy()
        edge_prob_np = edge_prob.detach().cpu().numpy().astype(np.float64)
        affinity_np = affinity.detach().cpu().numpy().astype(np.float64)
        seed_logits_np = seed_logits.detach().cpu().numpy().astype(np.float64)
        centrality_np = graph["centrality"].detach().cpu().numpy().astype(np.float64)

        downstream = affinity_np.copy()
        for _ in range(max(1, self.config.diffusion_steps - 1)):
            propagated = np.zeros_like(downstream)
            np.add.at(propagated, edge_src, edge_prob_np * downstream[edge_dst])
            downstream = affinity_np + propagated
        downstream = downstream / max(float(downstream.max()), 1e-8)

        k = self.dataset.config.k_seeds
        candidate_budget = min(affinity_np.shape[0], max(k * 6, 128))
        seed_rank = np.argsort(seed_logits_np)[-candidate_budget:]
        spread_rank = np.argsort(downstream + 0.10 * centrality_np)[-candidate_budget:]
        candidates = np.unique(np.concatenate([seed_rank, spread_rank]))
        ordering = np.argsort(seed_logits_np[candidates] + 0.45 * downstream[candidates] + 0.05 * centrality_np[candidates])[::-1]
        candidates = candidates[ordering][:candidate_budget]

        return self._coverage_aware_seed_selection(
            candidates=candidates,
            affinity=affinity_np,
            downstream=downstream,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_prob=edge_prob_np,
            centrality=centrality_np,
            k=k,
        )

    def _diffusion_bonus(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        src, dst = graph["edge_index"]
        affinity_logits = self._affinity_logits(user_repr, item_repr)
        affinity = torch.sigmoid(affinity_logits)
        seed_logits = self._seed_logits(user_repr, item_repr, graph, affinity_logits)
        activations = continuous_seed_probabilities(seed_logits, self.dataset.config.k_seeds)
        edge_prob = self._edge_probabilities(user_repr, item_repr, graph, affinity_logits, affinity)
        hidden = activations
        for _ in range(self.config.diffusion_steps):
            safe_term = torch.clamp(1.0 - edge_prob * hidden[src], min=1e-6, max=1.0)
            log_prod = torch.zeros(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
            log_prod.index_add_(0, dst, safe_term.log())
            hidden = 1.0 - (1.0 - hidden) * log_prod.exp()
        return hidden.mean()

    def compute_loss(
        self,
        graph: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        user_repr = self.encode_users(graph)
        item_repr = self.item_representations()
        rec_loss, metrics = self._recommendation_loss(
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
        spreads = [
            self._diffusion_bonus(user_repr=user_repr, item_repr=item_repr[item_idx], graph=graph)
            for item_idx in unique_items
        ]
        spread_bonus = torch.stack(spreads).mean() if spreads else torch.tensor(0.0, device=user_repr.device)
        lambda_scale = min(1.0, float(epoch + 1) / max(1, self.config.warmup_epochs))
        loss = rec_loss - (self.config.lambda_spread * lambda_scale * spread_bonus)
        metrics["spread_bonus"] = float(spread_bonus.detach().item())
        return loss, metrics

    def seed_scores(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        affinity_logits = self._affinity_logits(user_repr, item_repr)
        return self._seed_logits(user_repr, item_repr, graph, affinity_logits)

    def select_seeds(
        self,
        graph: dict[str, torch.Tensor],
        item_idx: int,
        user_repr: torch.Tensor | None = None,
    ) -> list[int]:
        with torch.no_grad():
            encoded = user_repr if user_repr is not None else self.encode_users(graph)
            item_repr = self.item_representations()[item_idx]
            affinity_logits = self._affinity_logits(encoded, item_repr)
            affinity = torch.sigmoid(affinity_logits)
            seed_logits = self._seed_logits(encoded, item_repr, graph, affinity_logits)
            edge_prob = self._edge_probabilities(encoded, item_repr, graph, affinity_logits, affinity)
        return self._graph_aware_seeds(seed_logits=seed_logits, affinity=affinity, graph=graph, edge_prob=edge_prob)


class TorchJRSOSecond(TorchJRSO):
    model_name = "JRSO_second"

    def __init__(self, dataset: SparseSyntheticDataset, config: TorchTrainingConfig) -> None:
        super().__init__(dataset, config)
        hidden_dim = config.hidden_dim
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.seed_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.src_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dst_projection = nn.Linear(hidden_dim, hidden_dim)
        self.item_projection = nn.Linear(hidden_dim, hidden_dim)

    def _recommendation_loss(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pos_scores = (user_repr[users] * item_repr[pos_items]).sum(dim=-1) / np.sqrt(user_repr.shape[1])
        candidate_scores = self._candidate_scores(user_repr, item_repr, users, neg_candidates)
        logits = torch.cat([pos_scores.unsqueeze(1), candidate_scores], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        metrics = {
            "rec_loss": float(loss.detach().item()),
            "pos_score": float(pos_scores.detach().mean().item()),
            "neg_score": float(candidate_scores.detach().mean().item()),
        }
        return loss, metrics

    def _affinity_logits(self, user_repr: torch.Tensor, item_repr: torch.Tensor) -> torch.Tensor:
        item_expanded = item_repr.unsqueeze(0).expand_as(user_repr)
        features = torch.cat(
            [user_repr, item_expanded, user_repr * item_expanded],
            dim=-1,
        )
        return self.affinity_head(features).squeeze(-1)

    def _seed_logits(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        graph: dict[str, torch.Tensor],
        affinity_logits: torch.Tensor,
    ) -> torch.Tensor:
        item_expanded = item_repr.unsqueeze(0).expand_as(user_repr)
        centrality_tensor = graph["centrality"].to(dtype=user_repr.dtype).unsqueeze(1)
        seed_features = torch.cat(
            [user_repr, item_expanded, user_repr * item_expanded, torch.abs(user_repr - item_expanded), centrality_tensor],
            dim=-1,
        )
        return self.seed_head(seed_features).squeeze(-1) + 0.35 * affinity_logits + self.seed_bias

    def _edge_probabilities(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        graph: dict[str, torch.Tensor],
        affinity_logits: torch.Tensor,
        affinity: torch.Tensor,
    ) -> torch.Tensor:
        del affinity_logits
        src, dst = graph["edge_index"]
        src_repr = self.src_projection(user_repr[src])
        dst_repr = self.dst_projection(user_repr[dst])
        item_context = self.item_projection(item_repr)
        edge_logits = (
            (src_repr * dst_repr).sum(dim=-1)
            + (src_repr * item_context.unsqueeze(0)).sum(dim=-1)
            + (dst_repr * item_context.unsqueeze(0)).sum(dim=-1)
        ) / np.sqrt(user_repr.shape[1])
        edge_prob = graph["edge_weight"] * torch.sigmoid(edge_logits) * affinity[src] * affinity[dst]
        return edge_prob.clamp(max=0.95)

    def compute_loss(
        self,
        graph: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        user_repr = self.encode_users(graph)
        item_repr = self.item_representations()
        rec_loss, metrics = self._recommendation_loss(
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
        spreads = [
            self._diffusion_bonus(user_repr=user_repr, item_repr=item_repr[item_idx], graph=graph)
            for item_idx in unique_items
        ]
        spread_bonus = torch.stack(spreads).mean() if spreads else torch.tensor(0.0, device=user_repr.device)
        lambda_scale = min(1.0, float(epoch + 1) / max(1, self.config.warmup_epochs))

        src, dst = graph["edge_index"]
        sample_size = min(src.shape[0], 16_384)
        sampled_edges = torch.randperm(src.shape[0], device=src.device)[:sample_size]
        edge_src = src[sampled_edges]
        edge_dst = dst[sampled_edges]
        edge_prior_logits = (
            self.src_projection(user_repr[edge_src]) * self.dst_projection(user_repr[edge_dst])
        ).sum(dim=-1) / np.sqrt(user_repr.shape[1])
        edge_alignment = F.mse_loss(torch.sigmoid(edge_prior_logits), graph["edge_weight"][sampled_edges])

        loss = rec_loss - (self.config.lambda_spread * lambda_scale * spread_bonus) + 0.08 * edge_alignment
        metrics["spread_bonus"] = float(spread_bonus.detach().item())
        metrics["edge_alignment"] = float(edge_alignment.detach().item())
        return loss, metrics


class InfluenceOnlyBaseline:
    model_name = "IM-Only"

    def __init__(self, dataset: SparseSyntheticDataset) -> None:
        self.dataset = dataset
        self.global_seeds = self._build_global_seeds()
        self.artifacts = TorchModelArtifacts(history=[], metadata={"global_seeds": self.global_seeds})

    def _build_global_seeds(self) -> list[int]:
        scores = self.dataset.centrality.copy()
        src, dst = self.dataset.edge_index
        selected: list[int] = []
        for _ in range(self.dataset.config.k_seeds):
            best_idx = int(np.argmax(scores))
            selected.append(best_idx)
            scores[best_idx] = -np.inf
            neighbor_mask = src == best_idx
            scores[dst[neighbor_mask]] *= 0.90
        return selected

    def select_seeds(self, item_idx: int) -> list[int]:
        del item_idx
        return list(self.global_seeds)


class TwoStageBaseline:
    model_name = "Two-Stage"

    def __init__(
        self,
        dataset: SparseSyntheticDataset,
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
    ) -> list[int]:
        with torch.no_grad():
            encoded = user_repr if user_repr is not None else self.recommender.encode_users(self.graph)
            item_repr = self.recommender.item_representations()[item_idx]
            affinity = torch.sigmoid((encoded * item_repr.unsqueeze(0)).sum(dim=-1) / np.sqrt(encoded.shape[1]))
            src, dst = self.graph["edge_index"]
            propagation_bonus = torch.zeros(encoded.shape[0], device=encoded.device)
            propagation_bonus.index_add_(0, src, self.graph["edge_weight"] * affinity[dst])
            seed_scores = affinity * (1.0 + propagation_bonus + self.graph["centrality"])
            _, indices = torch.topk(seed_scores, k=self.dataset.config.k_seeds)
        return [int(value) for value in indices.detach().cpu().tolist()]
