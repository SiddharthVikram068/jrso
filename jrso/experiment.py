from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .dataset import SyntheticDatasetGenerator
from .metrics import combined_score, influence_metrics, paired_permutation_test, pareto_frontier, recommendation_metrics
from .models import InfluenceMaximizationOnly, JRSOModel, RecSysOnlyModel, TwoStagePipeline
from .plots import save_metric_bar_chart, save_pareto_svg, save_seed_selection_svg, save_training_curve_svg


def _model_seeds(model: Any, evaluation_items: np.ndarray) -> dict[int, list[int]]:
    return {int(item_idx): list(map(int, model.select_seeds(int(item_idx)))) for item_idx in evaluation_items}


def _evaluate_trainable_model(model: Any, dataset: Any, recommendation_k: tuple[int, ...], seed: int) -> dict[str, Any]:
    score_matrix = model.predict_score_matrix()
    rec = recommendation_metrics(
        score_matrix=score_matrix,
        train_matrix=dataset.train_matrix,
        target_matrix=dataset.test_matrix,
        k_values=recommendation_k,
    )
    seeds = _model_seeds(model, dataset.evaluation_items)
    influence, per_item = influence_metrics(dataset, seeds, rng_seed=seed)
    return {
        "recommendation": rec,
        "influence": influence,
        "per_item": per_item,
        "seeds": seeds,
        "combined_score": combined_score(rec, influence),
        "history": model.artifacts.history,
    }


def _evaluate_im_only(model: InfluenceMaximizationOnly, dataset: Any, recommendation_k: tuple[int, ...], seed: int) -> dict[str, Any]:
    del recommendation_k
    rec = {"precision@5": 0.0, "recall@5": 0.0, "ndcg@5": 0.0, "precision@10": 0.0, "recall@10": 0.0, "ndcg@10": 0.0}
    seeds = _model_seeds(model, dataset.evaluation_items)
    influence, per_item = influence_metrics(dataset, seeds, rng_seed=seed)
    return {
        "recommendation": rec,
        "influence": influence,
        "per_item": per_item,
        "seeds": seeds,
        "combined_score": combined_score(rec, influence),
        "history": model.artifacts.history,
    }


def _write_metrics_csv(results: dict[str, dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model",
        "combined_score",
        "precision@5",
        "recall@5",
        "ndcg@5",
        "precision@10",
        "recall@10",
        "ndcg@10",
        "spread",
        "spread_ratio",
        "oracle_regret",
        "seed_precision",
        "adoption_jaccard",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, payload in results.items():
            row = {
                "model": model_name,
                "combined_score": payload["combined_score"],
            }
            row.update(payload["recommendation"])
            row.update(payload["influence"])
            writer.writerow(row)


def run_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    config = config or ExperimentConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticDatasetGenerator(config.dataset).generate()
    dataset.save(output_dir)

    recsys_model = RecSysOnlyModel(dataset, config.training, seed_offset=1).fit()
    two_stage_model = TwoStagePipeline(recommender=recsys_model, dataset=dataset)
    im_only_model = InfluenceMaximizationOnly(dataset)
    jrso_model = JRSOModel(dataset, config.training, seed_offset=2).fit()

    results = {
        recsys_model.model_name: _evaluate_trainable_model(recsys_model, dataset, config.recommendation_k, config.seed + 1),
        im_only_model.model_name: _evaluate_im_only(im_only_model, dataset, config.recommendation_k, config.seed + 2),
        two_stage_model.model_name: _evaluate_trainable_model(two_stage_model, dataset, config.recommendation_k, config.seed + 3),
        jrso_model.model_name: _evaluate_trainable_model(jrso_model, dataset, config.recommendation_k, config.seed + 4),
    }

    pareto_points: dict[str, tuple[float, float]] = {}
    lambda_histories: dict[str, list[dict[str, float]]] = {}
    for lambda_value in config.lambda_sweep:
        label = f"JRSO(lambda={lambda_value:.2f})"
        if abs(lambda_value - config.training.lambda_spread) < 1e-9:
            model = jrso_model
        else:
            model = JRSOModel(dataset, config.training.with_lambda(lambda_value), seed_offset=int(lambda_value * 10_000) + 3).fit()
        evaluation = _evaluate_trainable_model(model, dataset, config.recommendation_k, config.seed + int(lambda_value * 1_000) + 100)
        pareto_points[label] = (
            evaluation["recommendation"].get("ndcg@10", 0.0),
            evaluation["influence"].get("spread_ratio", 0.0),
        )
        lambda_histories[label] = evaluation["history"]

    frontier = pareto_frontier(pareto_points)
    significance = paired_permutation_test(
        results["JRSO"]["per_item"]["spread_ratio"],
        results["Two-Stage"]["per_item"]["spread_ratio"],
        seed=config.seed + 500,
    )

    history_payload = {
        "RecSys-Only": recsys_model.artifacts.history,
        "JRSO": jrso_model.artifacts.history,
        "Two-Stage": two_stage_model.artifacts.history,
    }
    history_payload.update(lambda_histories)
    (output_dir / "training_logs.json").write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    (output_dir / "pareto_frontier.json").write_text(
        json.dumps({"points": pareto_points, "frontier": frontier}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "significance.json").write_text(json.dumps(significance, indent=2), encoding="utf-8")

    compact_results = {
        model_name: {
            "combined_score": payload["combined_score"],
            **payload["recommendation"],
            **payload["influence"],
        }
        for model_name, payload in results.items()
    }
    _write_metrics_csv(results, output_dir / "metrics.csv")

    save_training_curve_svg(history_payload, output_dir / "training_curves.svg")
    save_metric_bar_chart(compact_results, "ndcg@10", output_dir / "recommendation_comparison.svg")
    save_metric_bar_chart(compact_results, "spread_ratio", output_dir / "influence_comparison.svg")
    save_pareto_svg(pareto_points, frontier, output_dir / "pareto_frontier.svg")

    focus_item = int(dataset.evaluation_items[0])
    save_seed_selection_svg(
        communities=dataset.communities,
        adjacency=dataset.social_adjacency,
        predicted_seeds=results["JRSO"]["seeds"][focus_item],
        oracle_seeds=dataset.oracle_seeds[focus_item],
        path=output_dir / f"seed_selection_item_{focus_item}.svg",
    )

    return {
        "config": config.to_dict(),
        "results": compact_results,
        "pareto_points": pareto_points,
        "pareto_frontier": frontier,
        "significance": significance,
        "output_dir": str(output_dir),
    }
