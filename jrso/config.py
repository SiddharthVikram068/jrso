from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for the synthetic JRSO benchmark."""

    num_users: int = 72
    num_items: int = 54
    embedding_dim: int = 10
    num_communities: int = 4
    intra_community_edge_prob: float = 0.20
    inter_community_edge_prob: float = 0.03
    interaction_density: float = 0.14
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    k_seeds: int = 5
    cascade_steps: int = 4
    influence_scale: float = 1.15
    num_eval_items: int = 6
    oracle_simulations: int = 48
    evaluation_simulations: int = 32
    seed: int = 7

    @classmethod
    def preset(cls, name: str) -> "DatasetConfig":
        presets = {
            "tiny": cls(
                num_users=24,
                num_items=18,
                embedding_dim=6,
                num_communities=3,
                intra_community_edge_prob=0.26,
                inter_community_edge_prob=0.04,
                interaction_density=0.18,
                k_seeds=3,
                cascade_steps=3,
                num_eval_items=4,
                oracle_simulations=24,
                evaluation_simulations=16,
                seed=3,
            ),
            "small": cls(),
            "medium": cls(
                num_users=120,
                num_items=90,
                embedding_dim=14,
                num_communities=5,
                intra_community_edge_prob=0.16,
                inter_community_edge_prob=0.02,
                interaction_density=0.10,
                k_seeds=6,
                cascade_steps=5,
                num_eval_items=8,
                oracle_simulations=64,
                evaluation_simulations=48,
                seed=11,
            ),
        }
        try:
            return presets[name]
        except KeyError as exc:
            raise ValueError(f"Unknown dataset preset: {name}") from exc

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration for recommender and joint models."""

    embedding_dim: int = 10
    gnn_layers: int = 1
    learning_rate: float = 0.04
    epochs: int = 18
    batch_size: int = 48
    batches_per_epoch: int = 8
    social_regularization: float = 0.04
    weight_decay: float = 1e-4
    lambda_spread: float = 0.15
    diffusion_steps: int = 4
    seed: int = 19

    def with_lambda(self, value: float) -> "TrainingConfig":
        return TrainingConfig(
            embedding_dim=self.embedding_dim,
            gnn_layers=self.gnn_layers,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            social_regularization=self.social_regularization,
            weight_decay=self.weight_decay,
            lambda_spread=value,
            diffusion_steps=self.diffusion_steps,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lambda_sweep: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    recommendation_k: tuple[int, ...] = (5, 10)
    output_dir: Path = Path("outputs/latest")
    seed: int = 29

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload
