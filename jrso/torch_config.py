from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class TorchDatasetConfig:
    """Sparse synthetic benchmark configuration for GPU-scale experiments."""

    num_users: int = 8_000
    num_items: int = 1_500
    latent_dim: int = 48
    num_communities: int = 20
    avg_intra_degree: int = 18
    avg_inter_degree: int = 4
    interaction_density: float = 0.020
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

    hidden_dim: int = 96
    num_layers: int = 3
    dropout: float = 0.10
    learning_rate: float = 5e-3
    weight_decay: float = 1e-5
    epochs: int = 80
    steps_per_epoch: int = 30
    batch_size: int = 4_096
    negative_candidates: int = 32
    spread_items_per_step: int = 6
    lambda_spread: float = 0.10
    diffusion_steps: int = 4
    gradient_clip_norm: float = 5.0
    eval_batch_size: int = 1_024
    amp: bool = True
    device: str = "auto"
    use_hard_negatives: bool = True
    warmup_epochs: int = 12
    diffusion_edge_cap: int = 32_768
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
    jrso_third: TorchTrainingConfig
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
                num_layers=2,
                dropout=0.10,
                learning_rate=5e-3,
                epochs=4,
                steps_per_epoch=4,
                batch_size=192,
                negative_candidates=16,
                spread_items_per_step=3,
                lambda_spread=0.05,
                diffusion_steps=3,
                eval_batch_size=128,
                amp=False,
                use_hard_negatives=True,
                warmup_epochs=2,
                seed=401,
            )
            jrso = recsys.replace(lambda_spread=0.15, seed=402)
            jrso_second = recsys.replace(
                hidden_dim=64,
                num_layers=3,
                dropout=0.10,
                negative_candidates=20,
                lambda_spread=0.18,
                spread_items_per_step=3,
                warmup_epochs=2,
                seed=403,
            )
            jrso_third = recsys.replace(
                hidden_dim=64,
                num_layers=3,
                dropout=0.10,
                learning_rate=5e-3,
                negative_candidates=24,
                spread_items_per_step=3,
                lambda_spread=0.22,
                warmup_epochs=2,
                seed=404,
            )
        else:
            recsys = TorchTrainingConfig(
                hidden_dim=128,
                num_layers=3,
                dropout=0.10,
                learning_rate=4e-3,
                weight_decay=1e-5,
                epochs=120,
                steps_per_epoch=40,
                batch_size=4096,
                negative_candidates=64,
                spread_items_per_step=6,
                lambda_spread=0.10,
                diffusion_steps=4,
                gradient_clip_norm=5.0,
                eval_batch_size=1024,
                amp=True,
                use_hard_negatives=True,
                warmup_epochs=20,
                seed=701,
            )
            jrso = recsys.replace(
                lambda_spread=0.15,
                spread_items_per_step=6,
                warmup_epochs=24,
                seed=702,
            )
            jrso_second = recsys.replace(
                hidden_dim=160,
                num_layers=4,
                dropout=0.15,
                learning_rate=3e-3,
                negative_candidates=64,
                spread_items_per_step=8,
                lambda_spread=0.20,
                warmup_epochs=28,
                seed=703,
            )
            jrso_third = recsys.replace(
                hidden_dim=160,
                num_layers=4,
                dropout=0.15,
                learning_rate=3e-3,
                negative_candidates=64,
                spread_items_per_step=8,
                lambda_spread=0.25,
                warmup_epochs=28,
                seed=704,
            )
        return cls(
            dataset=dataset,
            recsys=recsys,
            jrso=jrso,
            jrso_second=jrso_second,
            jrso_third=jrso_third,
            output_dir=output_dir,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset.to_dict(),
            "recsys": self.recsys.to_dict(),
            "jrso": self.jrso.to_dict(),
            "jrso_second": self.jrso_second.to_dict(),
            "jrso_third": self.jrso_third.to_dict(),
            "output_dir": str(self.output_dir),
            "recommendation_k": list(self.recommendation_k),
            "combined_alpha": self.combined_alpha,
            "seed": self.seed,
        }
