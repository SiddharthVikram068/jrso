from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .ciao_config import CiaoExperimentConfig
from .ciao_experiment import run_ciao_experiment
from .config import DatasetConfig, ExperimentConfig, TrainingConfig
from .experiment import run_experiment
from .torch_config import TorchExperimentConfig
from .torch_experiment import run_torch_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the JRSO benchmark and baselines.")
    parser.add_argument("--backend", choices=["reference", "torch", "ciao"], default="reference")
    parser.add_argument("--preset", default="small")
    parser.add_argument("--output-dir", default="outputs/latest")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lambda-spread", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ciao-dir",
        default="ciao",
        help="Directory containing rating.mat and trustnetwork.mat (only used when --backend=ciao).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.backend == "reference":
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
    elif args.backend == "ciao":
        ciao_config = CiaoExperimentConfig.preset(
            args.preset,
            Path(args.output_dir),
            ciao_dir=Path(args.ciao_dir),
        )
        if args.epochs is not None:
            ciao_config = CiaoExperimentConfig(
                dataset=ciao_config.dataset,
                recsys=ciao_config.recsys.replace(epochs=args.epochs),
                jrso=ciao_config.jrso.replace(epochs=args.epochs),
                jrso_second=ciao_config.jrso_second.replace(epochs=args.epochs),
                output_dir=ciao_config.output_dir,
                recommendation_k=ciao_config.recommendation_k,
                combined_alpha=ciao_config.combined_alpha,
                seed=ciao_config.seed,
            )
        if args.lambda_spread is not None:
            ciao_config = CiaoExperimentConfig(
                dataset=ciao_config.dataset,
                recsys=ciao_config.recsys,
                jrso=ciao_config.jrso.replace(lambda_spread=args.lambda_spread),
                jrso_second=ciao_config.jrso_second.replace(lambda_spread=args.lambda_spread),
                output_dir=ciao_config.output_dir,
                recommendation_k=ciao_config.recommendation_k,
                combined_alpha=ciao_config.combined_alpha,
                seed=ciao_config.seed,
            )
        if args.device is not None:
            ciao_config = CiaoExperimentConfig(
                dataset=ciao_config.dataset,
                recsys=ciao_config.recsys.replace(device=args.device),
                jrso=ciao_config.jrso.replace(device=args.device),
                jrso_second=ciao_config.jrso_second.replace(device=args.device),
                output_dir=ciao_config.output_dir,
                recommendation_k=ciao_config.recommendation_k,
                combined_alpha=ciao_config.combined_alpha,
                seed=ciao_config.seed,
            )
        result = run_ciao_experiment(ciao_config)
    else:
        experiment_config = TorchExperimentConfig.preset(args.preset, Path(args.output_dir))
        if args.epochs is not None:
            experiment_config = TorchExperimentConfig(
                dataset=experiment_config.dataset,
                recsys=experiment_config.recsys.replace(epochs=args.epochs),
                jrso=experiment_config.jrso.replace(epochs=args.epochs),
                jrso_second=experiment_config.jrso_second.replace(epochs=args.epochs),
                output_dir=experiment_config.output_dir,
                recommendation_k=experiment_config.recommendation_k,
                combined_alpha=experiment_config.combined_alpha,
                seed=experiment_config.seed,
            )
        if args.lambda_spread is not None:
            experiment_config = TorchExperimentConfig(
                dataset=experiment_config.dataset,
                recsys=experiment_config.recsys,
                jrso=experiment_config.jrso.replace(lambda_spread=args.lambda_spread),
                jrso_second=experiment_config.jrso_second,
                output_dir=experiment_config.output_dir,
                recommendation_k=experiment_config.recommendation_k,
                combined_alpha=experiment_config.combined_alpha,
                seed=experiment_config.seed,
            )
        if args.device is not None:
            experiment_config = TorchExperimentConfig(
                dataset=experiment_config.dataset,
                recsys=experiment_config.recsys.replace(device=args.device),
                jrso=experiment_config.jrso.replace(device=args.device),
                jrso_second=experiment_config.jrso_second.replace(device=args.device),
                output_dir=experiment_config.output_dir,
                recommendation_k=experiment_config.recommendation_k,
                combined_alpha=experiment_config.combined_alpha,
                seed=experiment_config.seed,
            )
        result = run_torch_experiment(experiment_config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
