from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .dataset import SyntheticDataset
from .diffusion import estimate_expected_spread


def recommendation_metrics(
    score_matrix: np.ndarray,
    train_matrix: np.ndarray,
    target_matrix: np.ndarray,
    k_values: Iterable[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    eligible_users = np.flatnonzero(target_matrix.sum(axis=1) > 0)
    if eligible_users.size == 0:
        for k in k_values:
            metrics[f"precision@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        return metrics

    masked_scores = score_matrix.copy()
    masked_scores[train_matrix > 0] = -1e9
    for k in k_values:
        precisions: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        topk = np.argpartition(-masked_scores[eligible_users], kth=min(k, masked_scores.shape[1] - 1), axis=1)[
            :, :k
        ]
        for local_idx, user_idx in enumerate(eligible_users):
            predictions = topk[local_idx]
            ranking = predictions[np.argsort(-masked_scores[user_idx, predictions])]
            truth = set(np.flatnonzero(target_matrix[user_idx]))
            if not truth:
                continue
            hits = [1 if item in truth else 0 for item in ranking]
            hit_count = sum(hits)
            precisions.append(hit_count / k)
            recalls.append(hit_count / len(truth))
            dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
            idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(truth), k)))
            ndcgs.append(dcg / (idcg + 1e-8))
        metrics[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return metrics


def influence_metrics(
    dataset: SyntheticDataset,
    item_to_seeds: dict[int, list[int]],
    rng_seed: int,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    spreads: list[float] = []
    spread_ratios: list[float] = []
    regrets: list[float] = []
    seed_precisions: list[float] = []
    adoption_overlaps: list[float] = []

    for offset, item_idx in enumerate(dataset.evaluation_items):
        item_key = int(item_idx)
        seeds = item_to_seeds[item_key]
        spread, adoption_probabilities = estimate_expected_spread(
            item_scores=dataset.score_matrix[:, item_key],
            social_adjacency=dataset.social_adjacency,
            influence_probabilities=dataset.influence_probabilities,
            seeds=seeds,
            cascade_steps=dataset.config.cascade_steps,
            influence_scale=dataset.config.influence_scale,
            num_trials=dataset.config.evaluation_simulations,
            rng_seed=rng_seed + offset,
        )
        oracle_spread = dataset.oracle_spreads[item_key]
        oracle_seeds = dataset.oracle_seeds[item_key]
        oracle_adopters = dataset.oracle_adoption_probabilities[item_key]
        overlap_size = len(set(seeds) & set(oracle_seeds))
        top_k = max(1, min(len(seeds) * 2, adoption_probabilities.shape[0]))
        pred_top = set(np.argsort(adoption_probabilities)[-top_k:])
        oracle_top = set(np.argsort(oracle_adopters)[-top_k:])
        jaccard = len(pred_top & oracle_top) / max(1, len(pred_top | oracle_top))

        spreads.append(spread)
        spread_ratios.append(spread / (oracle_spread + 1e-8))
        regrets.append(oracle_spread - spread)
        seed_precisions.append(overlap_size / max(1, dataset.config.k_seeds))
        adoption_overlaps.append(jaccard)

    metrics = {
        "spread": float(np.mean(spreads)) if spreads else 0.0,
        "spread_ratio": float(np.mean(spread_ratios)) if spread_ratios else 0.0,
        "oracle_regret": float(np.mean(regrets)) if regrets else 0.0,
        "seed_precision": float(np.mean(seed_precisions)) if seed_precisions else 0.0,
        "adoption_jaccard": float(np.mean(adoption_overlaps)) if adoption_overlaps else 0.0,
    }
    per_item = {
        "spread": spreads,
        "spread_ratio": spread_ratios,
        "oracle_regret": regrets,
        "seed_precision": seed_precisions,
        "adoption_jaccard": adoption_overlaps,
    }
    return metrics, per_item


def combined_score(
    recommendation_result: dict[str, float],
    influence_result: dict[str, float],
    alpha: float = 0.5,
) -> float:
    rec_component = recommendation_result.get("ndcg@10", 0.0)
    influence_component = influence_result.get("spread_ratio", 0.0)
    return float(alpha * rec_component + (1.0 - alpha) * influence_component)


def pareto_frontier(points: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    frontier: dict[str, tuple[float, float]] = {}
    for label, point in points.items():
        dominated = False
        for other_label, other_point in points.items():
            if label == other_label:
                continue
            if (
                other_point[0] >= point[0]
                and other_point[1] >= point[1]
                and (other_point[0] > point[0] or other_point[1] > point[1])
            ):
                dominated = True
                break
        if not dominated:
            frontier[label] = point
    return frontier


def paired_permutation_test(
    first: list[float],
    second: list[float],
    num_trials: int = 2_000,
    seed: int = 0,
) -> dict[str, float | bool]:
    if len(first) != len(second):
        raise ValueError("Permutation test requires paired samples of equal length.")
    differences = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(num_trials):
        signs = rng.choice(np.array([-1.0, 1.0]), size=differences.shape[0])
        permuted = abs(float(np.mean(differences * signs)))
        if permuted >= observed:
            exceedances += 1
    p_value = (exceedances + 1) / (num_trials + 1)
    return {
        "observed_mean_difference": observed,
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
