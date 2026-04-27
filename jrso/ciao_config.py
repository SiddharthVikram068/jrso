"""Configuration for the real-data Ciao backend of JRSO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .torch_config import TorchTrainingConfig


@dataclass(frozen=True)
class CiaoDatasetConfig:
    """Real-data Ciao benchmark configuration.

    Field naming matches ``TorchDatasetConfig`` so the existing torch models
    and the simulator in ``torch_experiment`` can consume a ``CiaoDataset``
    without any model-side changes.
    """

    ciao_dir: Path = Path("ciao")
    rating_file: str = "rating.mat"
    trust_file: str = "trustnetwork.mat"
    rating_field: str = "rating"
    trust_field: str = "trustnetwork"

    min_rating: int = 1
    user_core: int = 5
    item_core: int = 5
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    min_train_per_user: int = 1

    edge_prob_smoothing: float = 1.0
    edge_prob_floor: float = 0.02
    edge_prob_ceiling: float = 0.55

    k_seeds: int = 25
    cascade_steps: int = 4
    influence_scale: float = 1.10
    num_eval_items: int = 12
    evaluation_simulations: int = 16

    score_pop_weight: float = 1.0
    score_neighbor_weight: float = 1.5
    score_category_weight: float = 0.6
    score_bias: float = -1.5

    seed: int = 17

    num_users: int = 0
    num_items: int = 0
    num_categories: int = 0

    @classmethod
    def preset(cls, name: str) -> "CiaoDatasetConfig":
        if name == "ciao_smoke":
            return cls(
                user_core=10,
                item_core=10,
                k_seeds=10,
                cascade_steps=3,
                num_eval_items=6,
                evaluation_simulations=4,
                seed=11,
            )
        if name == "ciao_medium":
            return cls(
                user_core=10,
                item_core=10,
                k_seeds=20,
                cascade_steps=3,
                num_eval_items=10,
                evaluation_simulations=6,
                seed=13,
            )
        if name == "ciao_full":
            return cls()
        raise ValueError(f"Unknown ciao dataset preset: {name}")

    def with_resolved_sizes(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
    ) -> "CiaoDatasetConfig":
        return replace(
            self,
            num_users=num_users,
            num_items=num_items,
            num_categories=num_categories,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ciao_dir"] = str(self.ciao_dir)
        return payload


@dataclass(frozen=True)
class CiaoExperimentConfig:
    """Top-level configuration for the Ciao backend."""

    dataset: CiaoDatasetConfig
    recsys: TorchTrainingConfig
    jrso: TorchTrainingConfig
    jrso_second: TorchTrainingConfig
    output_dir: Path
    recommendation_k: tuple[int, ...] = (10, 20)
    combined_alpha: float = 0.5
    seed: int = 2026

    @classmethod
    def preset(cls, name: str, output_dir: Path, ciao_dir: Path | None = None) -> "CiaoExperimentConfig":
        dataset = CiaoDatasetConfig.preset(name)
        if ciao_dir is not None:
            dataset = replace(dataset, ciao_dir=ciao_dir)
        if name == "ciao_smoke":
            recsys = TorchTrainingConfig(
                hidden_dim=48,
                num_layers=2,
                dropout=0.10,
                learning_rate=4e-3,
                weight_decay=1e-5,
                epochs=3,
                steps_per_epoch=4,
                batch_size=256,
                negative_candidates=8,
                spread_items_per_step=3,
                lambda_spread=0.10,
                diffusion_steps=3,
                eval_batch_size=256,
                amp=False,
                use_hard_negatives=True,
                warmup_epochs=1,
                seed=601,
            )
            jrso = recsys.replace(lambda_spread=0.30, seed=602)
            jrso_second = recsys.replace(
                hidden_dim=64,
                num_layers=3,
                dropout=0.15,
                negative_candidates=16,
                lambda_spread=0.40,
                spread_items_per_step=4,
                use_hard_negatives=True,
                warmup_epochs=1,
                seed=603,
            )
        elif name == "ciao_medium":
            recsys = TorchTrainingConfig(
                hidden_dim=64,
                num_layers=2,
                dropout=0.15,
                learning_rate=4e-3,
                weight_decay=1e-5,
                epochs=10,
                steps_per_epoch=8,
                batch_size=512,
                negative_candidates=12,
                spread_items_per_step=2,
                lambda_spread=0.10,
                diffusion_steps=3,
                gradient_clip_norm=5.0,
                eval_batch_size=512,
                amp=False,
                use_hard_negatives=True,
                warmup_epochs=2,
                seed=701,
            )
            jrso = recsys.replace(lambda_spread=0.45, seed=702)
            jrso_second = recsys.replace(
                hidden_dim=96,
                num_layers=3,
                dropout=0.20,
                negative_candidates=20,
                lambda_spread=0.55,
                spread_items_per_step=3,
                warmup_epochs=3,
                seed=703,
            )
        else:
            recsys = TorchTrainingConfig(
                hidden_dim=128,
                num_layers=3,
                dropout=0.20,
                learning_rate=2e-3,
                weight_decay=1e-5,
                epochs=40,
                steps_per_epoch=20,
                batch_size=2048,
                negative_candidates=20,
                spread_items_per_step=4,
                lambda_spread=0.10,
                diffusion_steps=4,
                gradient_clip_norm=5.0,
                eval_batch_size=1024,
                amp=True,
                use_hard_negatives=True,
                warmup_epochs=6,
                seed=801,
            )
            jrso = recsys.replace(lambda_spread=0.32, seed=802)
            jrso_second = recsys.replace(
                hidden_dim=160,
                num_layers=4,
                dropout=0.25,
                learning_rate=1.5e-3,
                negative_candidates=32,
                spread_items_per_step=6,
                lambda_spread=0.45,
                warmup_epochs=10,
                seed=803,
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
