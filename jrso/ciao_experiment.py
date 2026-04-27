"""End-to-end runner for JRSO on the real Ciao dataset.

Re-uses :mod:`jrso.torch_experiment` for training/evaluation because
:class:`jrso.ciao_dataset.CiaoDataset` exposes the same attributes as
:class:`jrso.torch_dataset.SparseSyntheticDataset`. Only the I/O and
orchestration are Ciao-specific here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ciao_config import CiaoExperimentConfig
from .ciao_dataset import CiaoDataset, CiaoDatasetLoader
from .metrics import paired_permutation_test, pareto_frontier
from .plots import save_metric_bar_chart, save_pareto_svg, save_training_curve_svg
from .torch_experiment import (
    _evaluate_influence_model,
    _evaluate_seed_only_model,
    _evaluate_trainable_model,
    _evaluate_two_stage_model,
    _train_model,
    resolve_device,
)
from .torch_models import (
    InfluenceOnlyBaseline,
    TorchJRSO,
    TorchJRSOSecond,
    TorchRecSysOnly,
    TwoStageBaseline,
)


def _write_metrics_csv(results: dict[str, dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model",
        "combined_score",
        "precision@10",
        "recall@10",
        "ndcg@10",
        "precision@20",
        "recall@20",
        "ndcg@20",
        "spread_mean",
        "spread_rate",
        "weighted_spread_mean",
        "weighted_spread_rate",
        "test_positive_recall",
        "seed_item_affinity",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, payload in results.items():
            row = {"model": model_name, "combined_score": payload["combined_score"]}
            row.update(payload["recommendation"])
            row.update(payload["influence"])
            writer.writerow(row)


def _summarize_uplift(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute uplift of JRSO_second vs the two single-objective baselines.

    The paper's central claim is that JRSO Pareto-dominates RecSys-only
    (recommendation quality) and IM-only (influence spread) simultaneously.
    """

    def safe_pct(numerator: float, denominator: float) -> float:
        if abs(denominator) < 1e-9:
            return 0.0
        return float((numerator - denominator) / abs(denominator))

    def metric(name: str, key: str, root: str) -> float:
        return float(results[name][root].get(key, 0.0))

    uplift: dict[str, dict[str, float]] = {}
    if "JRSO_second" in results and "RecSys-Only" in results:
        uplift["JRSO_second_vs_RecSys-Only"] = {
            "ndcg@10_pct": safe_pct(
                metric("JRSO_second", "ndcg@10", "recommendation"),
                metric("RecSys-Only", "ndcg@10", "recommendation"),
            ),
            "weighted_spread_rate_pct": safe_pct(
                metric("JRSO_second", "weighted_spread_rate", "influence"),
                metric("RecSys-Only", "weighted_spread_rate", "influence"),
            ),
            "combined_score_pct": safe_pct(
                results["JRSO_second"]["combined_score"],
                results["RecSys-Only"]["combined_score"],
            ),
        }
    if "JRSO_second" in results and "IM-Only" in results:
        uplift["JRSO_second_vs_IM-Only"] = {
            "ndcg@10_pct": safe_pct(
                metric("JRSO_second", "ndcg@10", "recommendation"),
                metric("IM-Only", "ndcg@10", "recommendation"),
            ),
            "weighted_spread_rate_pct": safe_pct(
                metric("JRSO_second", "weighted_spread_rate", "influence"),
                metric("IM-Only", "weighted_spread_rate", "influence"),
            ),
            "combined_score_pct": safe_pct(
                results["JRSO_second"]["combined_score"],
                results["IM-Only"]["combined_score"],
            ),
        }
    if "JRSO" in results and "RecSys-Only" in results:
        uplift["JRSO_vs_RecSys-Only"] = {
            "ndcg@10_pct": safe_pct(
                metric("JRSO", "ndcg@10", "recommendation"),
                metric("RecSys-Only", "ndcg@10", "recommendation"),
            ),
            "weighted_spread_rate_pct": safe_pct(
                metric("JRSO", "weighted_spread_rate", "influence"),
                metric("RecSys-Only", "weighted_spread_rate", "influence"),
            ),
            "combined_score_pct": safe_pct(
                results["JRSO"]["combined_score"],
                results["RecSys-Only"]["combined_score"],
            ),
        }
    return uplift


def _print_leaderboard(compact: dict[str, dict[str, float]]) -> None:
    header = (
        f"{'Model':<14} {'Recall@10':>10} {'NDCG@10':>10} {'Spread%':>9} "
        f"{'wSpread%':>10} {'Combined':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, payload in compact.items():
        print(
            f"{name:<14} "
            f"{payload.get('recall@10', 0.0):>10.4f} "
            f"{payload.get('ndcg@10', 0.0):>10.4f} "
            f"{100 * payload.get('spread_rate', 0.0):>8.2f}% "
            f"{100 * payload.get('weighted_spread_rate', 0.0):>9.2f}% "
            f"{payload.get('combined_score', 0.0):>10.4f}"
        )


def run_ciao_experiment(config: CiaoExperimentConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.recsys.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"[ciao] Loading dataset from {config.dataset.ciao_dir} ...")
    dataset: CiaoDataset = CiaoDatasetLoader(config.dataset).load()
    dataset.save(output_dir)
    print(
        f"[ciao] After {config.dataset.user_core}-core filter: "
        f"users={dataset.config.num_users:,} items={dataset.config.num_items:,} "
        f"train_pairs={dataset.train_pairs.shape[0]:,} "
        f"trust_edges={dataset.edge_index.shape[1]:,}"
    )
    graph = dataset.to_torch(device)

    print("[ciao] Training RecSys-Only ...")
    recsys_model = _train_model(
        TorchRecSysOnly(dataset, config.recsys), dataset, graph, config.recsys, output_dir
    )
    print("[ciao] Training JRSO ...")
    jrso_model = _train_model(
        TorchJRSO(dataset, config.jrso), dataset, graph, config.jrso, output_dir
    )
    print("[ciao] Training JRSO_second ...")
    jrso_second_model = _train_model(
        TorchJRSOSecond(dataset, config.jrso_second),
        dataset,
        graph,
        config.jrso_second,
        output_dir,
    )

    im_only_model = InfluenceOnlyBaseline(dataset)
    two_stage_model = TwoStageBaseline(dataset, recsys_model, graph)

    print("[ciao] Evaluating ...")
    results = {
        recsys_model.model_name: _evaluate_trainable_model(
            recsys_model, dataset, graph, config.recommendation_k,
            alpha=config.combined_alpha, device=device,
        ),
        im_only_model.model_name: _evaluate_seed_only_model(
            im_only_model, dataset, graph, config.recommendation_k,
            alpha=config.combined_alpha, device=device,
        ),
        two_stage_model.model_name: _evaluate_two_stage_model(
            two_stage_model, dataset, graph, config.recommendation_k,
            alpha=config.combined_alpha, device=device,
        ),
        jrso_model.model_name: _evaluate_trainable_model(
            jrso_model, dataset, graph, config.recommendation_k,
            alpha=config.combined_alpha, device=device,
        ),
        jrso_second_model.model_name: _evaluate_trainable_model(
            jrso_second_model, dataset, graph, config.recommendation_k,
            alpha=config.combined_alpha, device=device,
        ),
    }

    history_payload = {
        recsys_model.model_name: recsys_model.artifacts.history,
        jrso_model.model_name: jrso_model.artifacts.history,
        jrso_second_model.model_name: jrso_second_model.artifacts.history,
    }
    compact_results = {
        model_name: {
            "combined_score": payload["combined_score"],
            **payload["recommendation"],
            **payload["influence"],
        }
        for model_name, payload in results.items()
    }
    pareto_points = {
        model_name: (
            payload["recommendation"].get("ndcg@10", 0.0),
            payload["influence"].get("weighted_spread_rate", 0.0),
        )
        for model_name, payload in results.items()
    }
    frontier = pareto_frontier(pareto_points)
    significance = {
        "JRSO_second_vs_RecSys-Only_weighted_spread": paired_permutation_test(
            results["JRSO_second"]["per_item"]["weighted_spread_mean"],
            results["RecSys-Only"]["per_item"]["weighted_spread_mean"],
            seed=config.seed + 11,
        ),
        "JRSO_second_vs_IM-Only_weighted_spread": paired_permutation_test(
            results["JRSO_second"]["per_item"]["weighted_spread_mean"],
            results["IM-Only"]["per_item"]["weighted_spread_mean"],
            seed=config.seed + 17,
        ),
        "JRSO_vs_RecSys-Only_weighted_spread": paired_permutation_test(
            results["JRSO"]["per_item"]["weighted_spread_mean"],
            results["RecSys-Only"]["per_item"]["weighted_spread_mean"],
            seed=config.seed + 23,
        ),
    }
    uplift = _summarize_uplift(results)

    (output_dir / "ciao_training_logs.json").write_text(
        json.dumps(history_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ciao_results.json").write_text(
        json.dumps({"config": config.to_dict(), "results": compact_results, "uplift": uplift}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ciao_significance.json").write_text(
        json.dumps(significance, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ciao_seed_summary.json").write_text(
        json.dumps({name: payload["seeds"] for name, payload in results.items()}, indent=2),
        encoding="utf-8",
    )
    _write_metrics_csv(results, output_dir / "ciao_metrics.csv")

    save_training_curve_svg(history_payload, output_dir / "ciao_training_curves.svg")
    save_metric_bar_chart(compact_results, "ndcg@10", output_dir / "ciao_recommendation_comparison.svg")
    save_metric_bar_chart(compact_results, "weighted_spread_rate", output_dir / "ciao_influence_comparison.svg")
    save_pareto_svg(pareto_points, frontier, output_dir / "ciao_pareto_frontier.svg")

    print("\n[ciao] Leaderboard")
    _print_leaderboard(compact_results)
    if uplift:
        print("\n[ciao] Uplift summary")
        for label, payload in uplift.items():
            ndcg_pct = 100 * payload.get("ndcg@10_pct", 0.0)
            ws_pct = 100 * payload.get("weighted_spread_rate_pct", 0.0)
            comb_pct = 100 * payload.get("combined_score_pct", 0.0)
            print(
                f"  {label}: NDCG@10 {ndcg_pct:+.1f}%, "
                f"wSpread {ws_pct:+.1f}%, Combined {comb_pct:+.1f}%"
            )

    return {
        "backend": "ciao",
        "device": str(device),
        "config": config.to_dict(),
        "results": compact_results,
        "uplift": uplift,
        "pareto_frontier": frontier,
        "significance": significance,
        "output_dir": str(output_dir),
    }
