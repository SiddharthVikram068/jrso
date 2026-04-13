from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DatasetConfig, ExperimentConfig, TrainingConfig
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the JRSO benchmark and baselines.")
    parser.add_argument("--preset", choices=["tiny", "small", "medium"], default="small")
    parser.add_argument("--output-dir", default="outputs/latest")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lambda-spread", type=float, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dataset_config = DatasetConfig.preset(args.preset)
    training_config = TrainingConfig()
    if args.epochs is not None:
        training_config = TrainingConfig(
            embedding_dim=training_config.embedding_dim,
            gnn_layers=training_config.gnn_layers,
            learning_rate=training_config.learning_rate,
            epochs=args.epochs,
            batch_size=training_config.batch_size,
            batches_per_epoch=training_config.batches_per_epoch,
            social_regularization=training_config.social_regularization,
            weight_decay=training_config.weight_decay,
            lambda_spread=training_config.lambda_spread,
            diffusion_steps=training_config.diffusion_steps,
            seed=training_config.seed,
        )
    if args.lambda_spread is not None:
        training_config = training_config.with_lambda(args.lambda_spread)
    experiment_config = ExperimentConfig(
        dataset=dataset_config,
        training=training_config,
        output_dir=Path(args.output_dir),
    )
    result = run_experiment(experiment_config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
