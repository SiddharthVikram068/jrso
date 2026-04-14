from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class TorchDatasetConfig:
    """Sparse synthetic benchmark configuration for GPU-scale experiments."""

    num_users: int = 10_000
    num_items: int = 1_000
    latent_dim: int = 48
    num_communities: int = 20
    avg_intra_degree: int = 18
    avg_inter_degree: int = 4
    interaction_density: float = 0.018
    min_interactions_per_user: int = 8
    max_interactions_per_user: int = 30
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    k_seeds: int = 50
    cascade_steps: int = 5
    influence_scale: float = 1.30
    num_eval_items: int = 16
    evaluation_simulations: int = 8
    seed: int = 123

    @classmethod
    def preset(cls, name: str) -> "TorchDatasetConfig":
        presets = {
            "gpu_smoke": cls(
                num_users=256,
                num_items=96,
                latent_dim=24,
                num_communities=4,
                avg_intra_degree=8,
                avg_inter_degree=2,
                interaction_density=0.08,
                min_interactions_per_user=4,
                max_interactions_per_user=10,
                k_seeds=8,
                cascade_steps=3,
                num_eval_items=6,
                evaluation_simulations=3,
                seed=17,
            ),
            "large_gpu": cls(),
        }
        try:
            return presets[name]
        except KeyError as exc:
            raise ValueError(f"Unknown torch dataset preset: {name}") from exc

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TorchTrainingConfig:
    """Training configuration for sparse torch models."""

    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.20
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    epochs: int = 60
    steps_per_epoch: int = 20
    batch_size: int = 4_096
    negative_candidates: int = 24
    spread_items_per_step: int = 6
    lambda_spread: float = 0.15
    diffusion_steps: int = 5
    gradient_clip_norm: float = 5.0
    eval_batch_size: int = 1_024
    amp: bool = True
    device: str = "auto"
    use_hard_negatives: bool = True
    warmup_epochs: int = 8
    seed: int = 321

    def replace(self, **changes: object) -> "TorchTrainingConfig":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TorchExperimentConfig:
    """Top-level experiment configuration for the sparse torch backend."""

    dataset: TorchDatasetConfig
    recsys: TorchTrainingConfig
    jrso: TorchTrainingConfig
    jrso_second: TorchTrainingConfig
    output_dir: Path
    recommendation_k: tuple[int, ...] = (10, 20)
    combined_alpha: float = 0.5
    seed: int = 2026

    @classmethod
    def preset(cls, name: str, output_dir: Path) -> "TorchExperimentConfig":
        dataset = TorchDatasetConfig.preset(name)
        if name == "gpu_smoke":
            recsys = TorchTrainingConfig(
                hidden_dim=48,
                num_layers=3,
                dropout=0.10,
                learning_rate=3e-3,
                epochs=2,
                steps_per_epoch=3,
                batch_size=192,
                negative_candidates=12,
                spread_items_per_step=3,
                lambda_spread=0.10,
                diffusion_steps=3,
                eval_batch_size=128,
                amp=False,
                seed=401,
            )
            jrso = recsys.replace(lambda_spread=0.18, seed=402)
            jrso_second = recsys.replace(
                hidden_dim=64,
                num_layers=4,
                dropout=0.15,
                negative_candidates=20,
                lambda_spread=0.24,
                use_hard_negatives=True,
                warmup_epochs=1,
                seed=403,
            )
        else:
            recsys = TorchTrainingConfig(seed=701)
            jrso = recsys.replace(lambda_spread=0.18, seed=702)
            jrso_second = recsys.replace(
                hidden_dim=160,
                num_layers=6,
                dropout=0.25,
                learning_rate=1.5e-3,
                negative_candidates=40,
                spread_items_per_step=8,
                lambda_spread=0.26,
                warmup_epochs=12,
                seed=703,
            )
        return cls(
            dataset=dataset,
            recsys=recsys,
            jrso=jrso,
            jrso_second=jrso_second,
            output_dir=output_dir,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset.to_dict(),
            "recsys": self.recsys.to_dict(),
            "jrso": self.jrso.to_dict(),
            "jrso_second": self.jrso_second.to_dict(),
            "output_dir": str(self.output_dir),
            "recommendation_k": list(self.recommendation_k),
            "combined_alpha": self.combined_alpha,
            "seed": self.seed,
        }
