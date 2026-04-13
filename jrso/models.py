from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import autograd.numpy as anp
from autograd import grad
import numpy as np

from .config import TrainingConfig
from .dataset import SyntheticDataset
from .diffusion import celf_select_seeds, differentiable_diffusion, sigmoid
from .metrics import recommendation_metrics
from .optim import adam_step, initialize_adam


def _copy_params(parameters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in parameters.items()}


@dataclass
class ModelArtifacts:
    history: list[dict[str, float]]
    metadata: dict[str, Any]


class BaseTrainableModel:
    """Shared latent-factor + social propagation base model."""

    model_name = "base"

    def __init__(self, dataset: SyntheticDataset, config: TrainingConfig, seed_offset: int = 0) -> None:
        self.dataset = dataset
        self.config = config
        self.rng = np.random.default_rng(config.seed + seed_offset)
        self.adjacency = dataset.social_adjacency.astype(np.float64)
        degree = self.adjacency.sum(axis=1, keepdims=True)
        self.normalized_adjacency = self.adjacency / np.clip(degree, 1.0, None)
        self.social_edge_count = float(max(self.adjacency.sum(), 1.0))
        self.train_positives = [np.flatnonzero(dataset.train_matrix[user_idx]) for user_idx in range(dataset.train_matrix.shape[0])]
        self.trainable_users = np.array(
            [idx for idx, positives in enumerate(self.train_positives) if positives.size > 0],
            dtype=np.int64,
        )
        self.parameters = self._initialize_parameters()
        self.artifacts = ModelArtifacts(history=[], metadata={})

    def _initialize_parameters(self) -> dict[str, np.ndarray]:
        dim = self.config.embedding_dim
        layer_shape = (self.config.gnn_layers, dim, dim)
        self_weights = np.stack(
            [np.eye(dim) + self.rng.normal(0.0, 0.05, size=(dim, dim)) for _ in range(self.config.gnn_layers)],
            axis=0,
        )
        neighbor_weights = np.stack(
            [np.eye(dim) + self.rng.normal(0.0, 0.05, size=(dim, dim)) for _ in range(self.config.gnn_layers)],
            axis=0,
        )
        return {
            "user_embeddings": self.rng.normal(0.0, 0.20, size=(self.dataset.config.num_users, dim)),
            "item_embeddings": self.rng.normal(0.0, 0.20, size=(self.dataset.config.num_items, dim)),
            "self_weights": self_weights.reshape(layer_shape),
            "neighbor_weights": neighbor_weights.reshape(layer_shape),
            "seed_bias": np.zeros(self.dataset.config.num_users, dtype=np.float64),
        }

    def _encode_users(self, parameters: dict[str, anp.ndarray]) -> anp.ndarray:
        hidden = parameters["user_embeddings"]
        for layer_idx in range(self.config.gnn_layers):
            self_term = hidden @ parameters["self_weights"][layer_idx]
            neighbor_term = self.normalized_adjacency @ hidden @ parameters["neighbor_weights"][layer_idx]
            hidden = anp.tanh(self_term + neighbor_term)
        return hidden

    def _score_matrix(self, parameters: dict[str, anp.ndarray]) -> anp.ndarray:
        user_embeddings = self._encode_users(parameters)
        return (user_embeddings @ parameters["item_embeddings"].T) / anp.sqrt(self.config.embedding_dim)

    def _bpr_loss(self, parameters: dict[str, anp.ndarray], batch: dict[str, np.ndarray]) -> anp.ndarray:
        user_embeddings = self._encode_users(parameters)
        item_embeddings = parameters["item_embeddings"]
        users = batch["users"]
        pos_items = batch["pos_items"]
        neg_items = batch["neg_items"]
        pos_scores = anp.sum(user_embeddings[users] * item_embeddings[pos_items], axis=1)
        neg_scores = anp.sum(user_embeddings[users] * item_embeddings[neg_items], axis=1)
        return -anp.mean(anp.log(sigmoid(pos_scores - neg_scores) + 1e-8))

    def _social_regularization(self, parameters: dict[str, anp.ndarray]) -> anp.ndarray:
        user_embeddings = self._encode_users(parameters)
        row_norms = anp.sum(user_embeddings**2, axis=1)
        pairwise_sq = row_norms[:, None] + row_norms[None, :] - 2.0 * (user_embeddings @ user_embeddings.T)
        return anp.sum(self.adjacency * pairwise_sq) / self.social_edge_count

    def _weight_decay(self, parameters: dict[str, anp.ndarray]) -> anp.ndarray:
        penalty = 0.0
        for name, value in parameters.items():
            if name == "seed_bias":
                penalty += 0.1 * anp.sum(value**2)
            else:
                penalty += anp.sum(value**2)
        return penalty

    def _spread_regularizer(
        self,
        parameters: dict[str, anp.ndarray],
        batch: dict[str, np.ndarray],
    ) -> anp.ndarray:
        return anp.array(0.0)

    def _objective(self, parameters: dict[str, anp.ndarray], batch: dict[str, np.ndarray]) -> anp.ndarray:
        bpr = self._bpr_loss(parameters, batch)
        social = self.config.social_regularization * self._social_regularization(parameters)
        decay = self.config.weight_decay * self._weight_decay(parameters)
        spread = self._spread_regularizer(parameters, batch)
        return bpr + social + decay - spread

    def _objective_components(
        self,
        parameters: dict[str, np.ndarray],
        batch: dict[str, np.ndarray],
    ) -> dict[str, float]:
        bpr = float(self._bpr_loss(parameters, batch))
        social = float(self.config.social_regularization * self._social_regularization(parameters))
        decay = float(self.config.weight_decay * self._weight_decay(parameters))
        spread = float(self._spread_regularizer(parameters, batch))
        total = bpr + social + decay - spread
        return {
            "objective": total,
            "bpr_loss": bpr,
            "social_regularization": social,
            "weight_decay": decay,
            "spread_bonus": spread,
        }

    def _sample_batch(self) -> dict[str, np.ndarray]:
        users = self.rng.choice(self.trainable_users, size=self.config.batch_size, replace=True)
        pos_items = np.empty(self.config.batch_size, dtype=np.int64)
        neg_items = np.empty(self.config.batch_size, dtype=np.int64)
        num_items = self.dataset.train_matrix.shape[1]
        for idx, user_idx in enumerate(users):
            pos_items[idx] = int(self.rng.choice(self.train_positives[user_idx]))
            while True:
                candidate = int(self.rng.integers(0, num_items))
                if self.dataset.train_matrix[user_idx, candidate] == 0:
                    neg_items[idx] = candidate
                    break
        return {
            "users": users,
            "pos_items": pos_items,
            "neg_items": neg_items,
            "unique_pos_items": np.unique(pos_items),
        }

    def fit(self) -> "BaseTrainableModel":
        objective_grad = grad(self._objective)
        adam_state = initialize_adam(self.parameters)
        best_params = _copy_params(self.parameters)
        best_metric = -np.inf

        for epoch in range(self.config.epochs):
            batch_logs: list[dict[str, float]] = []
            for _ in range(self.config.batches_per_epoch):
                batch = self._sample_batch()
                gradients = objective_grad(self.parameters, batch)
                self.parameters, adam_state = adam_step(
                    parameters=self.parameters,
                    gradients=gradients,
                    state=adam_state,
                    learning_rate=self.config.learning_rate,
                )
                batch_logs.append(self._objective_components(self.parameters, batch))

            validation_scores = self.predict_score_matrix()
            validation_metrics = recommendation_metrics(
                score_matrix=validation_scores,
                train_matrix=self.dataset.train_matrix,
                target_matrix=self.dataset.val_matrix,
                k_values=(10,),
            )
            epoch_metrics = {
                "epoch": float(epoch),
                "objective": float(np.mean([entry["objective"] for entry in batch_logs])),
                "bpr_loss": float(np.mean([entry["bpr_loss"] for entry in batch_logs])),
                "spread_bonus": float(np.mean([entry["spread_bonus"] for entry in batch_logs])),
                "val_ndcg@10": float(validation_metrics["ndcg@10"]),
            }
            self.artifacts.history.append(epoch_metrics)
            if epoch_metrics["val_ndcg@10"] >= best_metric:
                best_metric = epoch_metrics["val_ndcg@10"]
                best_params = _copy_params(self.parameters)

        self.parameters = best_params
        self.artifacts.metadata["best_val_ndcg@10"] = best_metric
        return self

    def encoded_user_embeddings(self) -> np.ndarray:
        return np.asarray(self._encode_users(self.parameters), dtype=np.float64)

    def predict_score_matrix(self) -> np.ndarray:
        return np.asarray(self._score_matrix(self.parameters), dtype=np.float64)

    def select_seeds(self, item_idx: int) -> list[int]:
        raise NotImplementedError


class RecSysOnlyModel(BaseTrainableModel):
    model_name = "RecSys-Only"

    def select_seeds(self, item_idx: int) -> list[int]:
        scores = self.predict_score_matrix()[:, item_idx]
        top = np.argsort(scores)[-self.dataset.config.k_seeds :]
        return list(map(int, top[::-1]))


class JRSOModel(BaseTrainableModel):
    model_name = "JRSO"

    def _spread_regularizer(
        self,
        parameters: dict[str, anp.ndarray],
        batch: dict[str, np.ndarray],
    ) -> anp.ndarray:
        user_embeddings = self._encode_users(parameters)
        spread_terms = []
        for item_idx in batch["unique_pos_items"]:
            spread, _, _ = differentiable_diffusion(
                user_embeddings=user_embeddings,
                item_embedding=parameters["item_embeddings"][item_idx],
                social_adjacency=self.adjacency,
                seed_bias=parameters["seed_bias"],
                k_seeds=self.dataset.config.k_seeds,
                cascade_steps=self.config.diffusion_steps,
            )
            spread_terms.append(spread)
        if not spread_terms:
            return anp.array(0.0)
        return self.config.lambda_spread * anp.mean(anp.stack(spread_terms))

    def select_seeds(self, item_idx: int) -> list[int]:
        user_embeddings = self.encoded_user_embeddings()
        item_embedding = self.parameters["item_embeddings"][item_idx]
        affinity = np.asarray(sigmoid((user_embeddings @ item_embedding) / np.sqrt(self.config.embedding_dim)))
        seed_scores = affinity + self.parameters["seed_bias"]
        top = np.argsort(seed_scores)[-self.dataset.config.k_seeds :]
        return list(map(int, top[::-1]))


class InfluenceMaximizationOnly:
    model_name = "IM-Only"

    def __init__(self, dataset: SyntheticDataset) -> None:
        self.dataset = dataset
        average_scores = dataset.score_matrix.mean(axis=1)
        self.global_seeds = celf_select_seeds(
            item_scores=average_scores,
            social_adjacency=dataset.social_adjacency,
            influence_probabilities=dataset.influence_probabilities,
            k=dataset.config.k_seeds,
            cascade_steps=dataset.config.cascade_steps,
            influence_scale=dataset.config.influence_scale,
            num_trials=dataset.config.oracle_simulations,
            rng_seed=dataset.config.seed + 20_000,
        )
        self.artifacts = ModelArtifacts(history=[], metadata={"global_seeds": self.global_seeds})

    def select_seeds(self, item_idx: int) -> list[int]:
        del item_idx
        return list(self.global_seeds)


class TwoStagePipeline:
    model_name = "Two-Stage"

    def __init__(self, recommender: RecSysOnlyModel, dataset: SyntheticDataset) -> None:
        self.recommender = recommender
        self.dataset = dataset
        self.artifacts = ModelArtifacts(
            history=list(recommender.artifacts.history),
            metadata={"derived_from": recommender.model_name},
        )

    def select_seeds(self, item_idx: int) -> list[int]:
        user_embeddings = self.recommender.encoded_user_embeddings()
        item_embedding = self.recommender.parameters["item_embeddings"][item_idx]
        affinity = np.asarray(sigmoid((user_embeddings @ item_embedding) / np.sqrt(self.recommender.config.embedding_dim)))
        similarity = np.asarray(sigmoid((user_embeddings @ user_embeddings.T) / np.sqrt(self.recommender.config.embedding_dim)))
        estimated_influence = self.dataset.social_adjacency * similarity * affinity[:, None] * affinity[None, :]
        return celf_select_seeds(
            item_scores=affinity,
            social_adjacency=self.dataset.social_adjacency,
            influence_probabilities=estimated_influence,
            k=self.dataset.config.k_seeds,
            cascade_steps=self.dataset.config.cascade_steps,
            influence_scale=self.dataset.config.influence_scale,
            num_trials=self.dataset.config.evaluation_simulations,
            rng_seed=self.dataset.config.seed + 30_000 + int(item_idx),
        )

    def predict_score_matrix(self) -> np.ndarray:
        return self.recommender.predict_score_matrix()
