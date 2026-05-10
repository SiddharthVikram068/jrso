"""End-to-end runner for the synthetic torch experiment.

This module also exposes the helper functions that the Ciao runner re-uses
(``_train_model``, ``_evaluate_trainable_model``, ``_evaluate_seed_only_model``,
``_evaluate_two_stage_model``, ``resolve_device``).
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .metrics import paired_permutation_test, pareto_frontier
from .plots import save_metric_bar_chart, save_pareto_svg, save_training_curve_svg
from .torch_config import TorchExperimentConfig, TorchTrainingConfig
from .torch_dataset import SparseSyntheticDataset, SparseSyntheticDatasetGenerator
from .torch_models import (
    BaseTorchRecommender,
    InfluenceOnlyBaseline,
    TorchJRSO,
    TorchJRSOSecond,
    TorchJRSOThird,
    TorchRecSysOnly,
    TwoStageBaseline,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return device


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------


def _build_user_positive_index(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(flat_positives, offsets)`` so that ``flat_positives[offsets[u]:offsets[u+1]]``
    lists the positive items for user ``u``.
    """

    counts = matrix.sum(axis=1).astype(np.int64)
    offsets = np.empty(matrix.shape[0] + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    flat = np.empty(int(offsets[-1]), dtype=np.int64)
    pos_users, pos_items = np.nonzero(matrix)
    sort_idx = np.argsort(pos_users, kind="stable")
    flat[:] = pos_items[sort_idx]
    return flat, offsets


def _sample_batch(
    dataset: SparseSyntheticDataset,
    flat_positives: np.ndarray,
    offsets: np.ndarray,
    train_csr,
    config: TorchTrainingConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    num_users = int(dataset.config.num_users)
    num_items = int(dataset.config.num_items)
    user_indices = rng.integers(0, num_users, size=config.batch_size)
    counts = offsets[user_indices + 1] - offsets[user_indices]
    has_pos = counts > 0
    positive_items = np.empty(config.batch_size, dtype=np.int64)
    if has_pos.any():
        chosen_offset = rng.integers(0, np.maximum(counts[has_pos], 1))
        positive_items[has_pos] = flat_positives[offsets[user_indices[has_pos]] + chosen_offset]
    if (~has_pos).any():
        positive_items[~has_pos] = rng.integers(0, num_items, size=int((~has_pos).sum()))

    # Vectorised negative sampling: sample with rejection.
    negative_pool = rng.integers(
        0, num_items, size=(config.batch_size, config.negative_candidates * 2)
    )
    flat_users = np.repeat(user_indices, config.negative_candidates * 2)
    flat_items = negative_pool.ravel()
    flat_is_positive = np.asarray(train_csr[flat_users, flat_items]).ravel().astype(bool)
    is_positive = flat_is_positive.reshape(config.batch_size, config.negative_candidates * 2)
    negative_pool[is_positive] = rng.integers(0, num_items, size=int(is_positive.sum()))
    negative_candidates = negative_pool[:, : config.negative_candidates]

    return {
        "users": torch.as_tensor(user_indices, dtype=torch.long, device=device),
        "pos_items": torch.as_tensor(positive_items, dtype=torch.long, device=device),
        "neg_candidates": torch.as_tensor(negative_candidates, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _recommendation_metrics(
    score_matrix: np.ndarray,
    train_matrix: np.ndarray,
    target_matrix: np.ndarray,
    k_values: tuple[int, ...],
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
        max_k = min(k, masked_scores.shape[1])
        partition = np.argpartition(-masked_scores[eligible_users], kth=max_k - 1, axis=1)[:, :max_k]
        partition_scores = np.take_along_axis(masked_scores[eligible_users], partition, axis=1)
        order = np.argsort(-partition_scores, axis=1)
        topk_indices = np.take_along_axis(partition, order, axis=1)

        # Build a boolean hit matrix: hits[i, r] = 1 if topk_indices[i, r] is a true positive.
        eligible_truth = target_matrix[eligible_users].astype(bool)
        rows = np.broadcast_to(np.arange(eligible_users.size)[:, None], topk_indices.shape)
        hits = eligible_truth[rows, topk_indices]
        hits = hits.astype(np.float32)

        ranks = np.arange(max_k, dtype=np.float32)
        discount = 1.0 / np.log2(ranks + 2.0)
        dcg = (hits * discount).sum(axis=1)

        truth_counts = eligible_truth.sum(axis=1).astype(np.float32)
        idcg = np.array(
            [discount[: int(min(int(t), max_k))].sum() for t in truth_counts],
            dtype=np.float32,
        )
        ndcgs = dcg / np.clip(idcg, 1e-8, None)

        precisions = hits.sum(axis=1) / float(max_k)
        recalls = hits.sum(axis=1) / np.clip(truth_counts, 1.0, None)

        metrics[f"precision@{k}"] = float(precisions.mean())
        metrics[f"recall@{k}"] = float(recalls.mean())
        metrics[f"ndcg@{k}"] = float(ndcgs.mean())
    return metrics


def _simulate_item_spread(
    dataset: SparseSyntheticDataset,
    graph: dict[str, torch.Tensor],
    item_idx: int,
    seeds: list[int],
    device: torch.device,
) -> dict[str, float]:
    item_scores = torch.as_tensor(dataset.score_matrix[:, item_idx], dtype=torch.float32, device=device)
    base_logits = torch.logit(item_scores.clamp(1e-4, 1.0 - 1e-4))
    src, dst = graph["edge_index"]
    edge_weight = graph["edge_weight"]
    num_users = dataset.config.num_users
    spread_values = np.empty(dataset.config.evaluation_simulations, dtype=np.float32)
    weighted_values = np.empty(dataset.config.evaluation_simulations, dtype=np.float32)
    positive_recall_values = np.empty(dataset.config.evaluation_simulations, dtype=np.float32)
    test_positive_mask = torch.as_tensor(dataset.test_matrix[:, item_idx], dtype=torch.bool, device=device)
    test_positive_count = int(test_positive_mask.sum().item())

    for sim in range(dataset.config.evaluation_simulations):
        active = torch.zeros(num_users, dtype=torch.bool, device=device)
        frontier = torch.zeros(num_users, dtype=torch.bool, device=device)
        active[seeds] = True
        frontier[seeds] = True
        for _step in range(dataset.config.cascade_steps):
            active_edge_mask = frontier[src]
            if not bool(active_edge_mask.any()):
                break
            candidate_dst = dst[active_edge_mask]
            candidate_weight = edge_weight[active_edge_mask]
            inactive_mask = ~active[candidate_dst]
            candidate_dst = candidate_dst[inactive_mask]
            candidate_weight = candidate_weight[inactive_mask]
            if candidate_dst.numel() == 0:
                frontier.zero_()
                break
            successful = torch.rand(candidate_weight.shape[0], device=device) < candidate_weight
            if not bool(successful.any()):
                frontier.zero_()
                break
            successful_dst = candidate_dst[successful]
            successful_weight = candidate_weight[successful]
            influence_sum = torch.zeros(num_users, dtype=torch.float32, device=device)
            influence_sum.index_add_(0, successful_dst, successful_weight)
            candidates = (~active) & (influence_sum > 0)
            if not bool(candidates.any()):
                frontier.zero_()
                break
            candidate_indices = torch.nonzero(candidates, as_tuple=False).squeeze(1)
            candidate_logits = (
                base_logits[candidate_indices]
                + dataset.config.influence_scale * influence_sum[candidate_indices]
            )
            candidate_probs = torch.sigmoid(candidate_logits)
            candidate_draws = torch.rand(candidate_probs.shape[0], device=device) < candidate_probs
            frontier = torch.zeros(num_users, dtype=torch.bool, device=device)
            if bool(candidate_draws.any()):
                frontier[candidate_indices[candidate_draws]] = True
                active |= frontier
            else:
                break
        spread_values[sim] = float(active.sum().item())
        weighted_values[sim] = float(item_scores[active].sum().item())
        if test_positive_count > 0:
            positive_recall_values[sim] = float(
                (active & test_positive_mask).sum().item() / test_positive_count
            )
        else:
            positive_recall_values[sim] = 0.0

    return {
        "spread_mean": float(spread_values.mean()),
        "spread_rate": float(spread_values.mean() / dataset.config.num_users),
        "weighted_spread_mean": float(weighted_values.mean()),
        "weighted_spread_rate": float(weighted_values.mean() / dataset.config.num_users),
        "test_positive_recall": float(positive_recall_values.mean()),
    }


def _evaluate_influence_model(
    dataset: SparseSyntheticDataset,
    graph: dict[str, torch.Tensor],
    seed_lookup: dict[int, list[int]],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    spreads: list[float] = []
    spread_rates: list[float] = []
    weighted_spreads: list[float] = []
    weighted_rates: list[float] = []
    positive_recalls: list[float] = []
    seed_quality: list[float] = []
    for item_idx in dataset.evaluation_items:
        item_key = int(item_idx)
        seeds = seed_lookup[item_key]
        item_metrics = _simulate_item_spread(dataset, graph, item_key, seeds, device=device)
        spreads.append(item_metrics["spread_mean"])
        spread_rates.append(item_metrics["spread_rate"])
        weighted_spreads.append(item_metrics["weighted_spread_mean"])
        weighted_rates.append(item_metrics["weighted_spread_rate"])
        positive_recalls.append(item_metrics["test_positive_recall"])
        seed_quality.append(float(dataset.score_matrix[seeds, item_key].mean()))
    summary = {
        "spread_mean": float(np.mean(spreads)) if spreads else 0.0,
        "spread_rate": float(np.mean(spread_rates)) if spread_rates else 0.0,
        "weighted_spread_mean": float(np.mean(weighted_spreads)) if weighted_spreads else 0.0,
        "weighted_spread_rate": float(np.mean(weighted_rates)) if weighted_rates else 0.0,
        "test_positive_recall": float(np.mean(positive_recalls)) if positive_recalls else 0.0,
        "seed_item_affinity": float(np.mean(seed_quality)) if seed_quality else 0.0,
    }
    per_item = {
        "spread_mean": spreads,
        "spread_rate": spread_rates,
        "weighted_spread_mean": weighted_spreads,
        "weighted_spread_rate": weighted_rates,
        "test_positive_recall": positive_recalls,
        "seed_item_affinity": seed_quality,
    }
    return summary, per_item


def _combined_score(
    recommendation_metrics: dict[str, float],
    influence_metrics: dict[str, float],
    alpha: float,
) -> float:
    return float(
        alpha * recommendation_metrics.get("ndcg@10", 0.0)
        + (1.0 - alpha) * influence_metrics.get("weighted_spread_rate", 0.0)
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
    def safe_pct(numerator: float, denominator: float) -> float:
        if abs(denominator) < 1e-9:
            return 0.0
        return float((numerator - denominator) / abs(denominator))

    def metric(name: str, key: str, root: str) -> float:
        return float(results[name][root].get(key, 0.0))

    uplift: dict[str, dict[str, float]] = {}
    for challenger in ("JRSO", "JRSO_second", "JRSO_third"):
        if challenger not in results:
            continue
        for opponent in ("RecSys-Only", "Two-Stage", "IM-Only"):
            if opponent not in results:
                continue
            uplift[f"{challenger}_vs_{opponent}"] = {
                "ndcg@10_pct": safe_pct(
                    metric(challenger, "ndcg@10", "recommendation"),
                    metric(opponent, "ndcg@10", "recommendation"),
                ),
                "weighted_spread_rate_pct": safe_pct(
                    metric(challenger, "weighted_spread_rate", "influence"),
                    metric(opponent, "weighted_spread_rate", "influence"),
                ),
                "combined_score_pct": safe_pct(
                    results[challenger]["combined_score"],
                    results[opponent]["combined_score"],
                ),
            }
    return uplift


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_model(
    model: BaseTorchRecommender,
    dataset: SparseSyntheticDataset,
    graph: dict[str, torch.Tensor],
    config: TorchTrainingConfig,
    output_dir: Path,
) -> BaseTorchRecommender:
    device = graph["edge_index"].device
    print(f"[train] {model.model_name}: device={device} epochs={config.epochs} steps={config.steps_per_epoch}")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=config.learning_rate * 0.10,
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    rng = np.random.default_rng(config.seed)

    import scipy.sparse as sp

    train_csr = sp.csr_matrix(dataset.train_matrix.astype(np.uint8))
    flat_positives, offsets = _build_user_positive_index(dataset.train_matrix)

    best_state = copy.deepcopy(model.state_dict())
    best_val_combined = float("-inf")
    best_val_ndcg = float("-inf")

    for epoch in range(config.epochs):
        model.train()
        epoch_logs: list[dict[str, float]] = []
        for _step in range(config.steps_per_epoch):
            batch = _sample_batch(dataset, flat_positives, offsets, train_csr, config, rng, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, metrics = model.compute_loss(graph, batch, epoch=epoch)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
            epoch_logs.append(
                {
                    "objective": float(loss.detach().item()),
                    "rec_loss": float(metrics["rec_loss"]),
                    "spread_bonus": float(metrics.get("spread_bonus", 0.0)),
                    "edge_alignment": float(metrics.get("edge_alignment", 0.0)),
                }
            )

        epoch_spread_bonus = float(np.mean([entry["spread_bonus"] for entry in epoch_logs]))
        eval_every = 2 if config.epochs <= 30 else 4
        should_eval = epoch == 0 or (epoch + 1) % eval_every == 0 or epoch == config.epochs - 1
        if should_eval:
            model.eval()
            with torch.no_grad():
                val_scores = model.predict_score_matrix(graph)
            val_metrics = _recommendation_metrics(
                score_matrix=val_scores,
                train_matrix=dataset.train_matrix,
                target_matrix=dataset.val_matrix,
                k_values=(10,),
            )
            val_ndcg = float(val_metrics["ndcg@10"])
            combined = val_ndcg + 0.20 * epoch_spread_bonus
            if combined > best_val_combined + 1e-6 or (
                abs(combined - best_val_combined) <= 1e-6 and val_ndcg > best_val_ndcg
            ):
                best_val_combined = combined
                best_val_ndcg = val_ndcg
                best_state = copy.deepcopy(model.state_dict())
        else:
            val_ndcg = best_val_ndcg

        model.artifacts.history.append(
            {
                "epoch": float(epoch),
                "objective": float(np.mean([entry["objective"] for entry in epoch_logs])),
                "rec_loss": float(np.mean([entry["rec_loss"] for entry in epoch_logs])),
                "spread_bonus": epoch_spread_bonus,
                "edge_alignment": float(np.mean([entry["edge_alignment"] for entry in epoch_logs])),
                "val_ndcg@10": float(val_ndcg if np.isfinite(val_ndcg) else 0.0),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if should_eval or epoch == config.epochs - 1:
            print(
                f"[train] {model.model_name} epoch {epoch + 1}/{config.epochs}: "
                f"rec_loss={float(np.mean([entry['rec_loss'] for entry in epoch_logs])):.4f} "
                f"spread_bonus={epoch_spread_bonus:.4f} "
                f"val_ndcg@10={float(val_ndcg if np.isfinite(val_ndcg) else 0.0):.4f}",
                flush=True,
            )
        scheduler.step()

    model.load_state_dict(best_state)
    model.artifacts.metadata["best_val_ndcg@10"] = float(
        best_val_ndcg if np.isfinite(best_val_ndcg) else 0.0
    )
    model.artifacts.metadata["best_val_combined"] = float(
        best_val_combined if np.isfinite(best_val_combined) else 0.0
    )
    checkpoint_path = output_dir / f"{model.model_name.lower().replace('-', '_')}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return model


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------


def _evaluate_trainable_model(
    model: BaseTorchRecommender,
    dataset: SparseSyntheticDataset,
    graph: dict[str, torch.Tensor],
    recommendation_k: tuple[int, ...],
    alpha: float,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        user_repr, item_repr = model.encode(graph)
        score_tensor = model.score_all(graph, user_repr=user_repr, item_repr=item_repr)
        score_matrix = score_tensor.detach().cpu().numpy()
    recommendation = _recommendation_metrics(
        score_matrix=score_matrix,
        train_matrix=dataset.train_matrix,
        target_matrix=dataset.test_matrix,
        k_values=recommendation_k,
    )
    seed_lookup = {
        int(item_idx): model.select_seeds(graph, int(item_idx), user_repr=user_repr, item_repr=item_repr)
        for item_idx in dataset.evaluation_items
    }
    influence, per_item = _evaluate_influence_model(dataset, graph, seed_lookup, device=device)
    return {
        "recommendation": recommendation,
        "influence": influence,
        "per_item": per_item,
        "seeds": seed_lookup,
        "combined_score": _combined_score(recommendation, influence, alpha=alpha),
        "history": model.artifacts.history,
    }


def _evaluate_seed_only_model(
    model: Any,
    dataset: SparseSyntheticDataset,
    graph: dict[str, torch.Tensor],
    recommendation_k: tuple[int, ...],
    alpha: float,
    device: torch.device,
) -> dict[str, Any]:
    del recommendation_k
    recommendation = {
        "precision@10": 0.0,
        "recall@10": 0.0,
        "ndcg@10": 0.0,
        "precision@20": 0.0,
        "recall@20": 0.0,
        "ndcg@20": 0.0,
    }
    seed_lookup = {
        int(item_idx): model.select_seeds(int(item_idx))
        for item_idx in dataset.evaluation_items
    }
    influence, per_item = _evaluate_influence_model(dataset, graph, seed_lookup, device=device)
    return {
        "recommendation": recommendation,
        "influence": influence,
        "per_item": per_item,
        "seeds": seed_lookup,
        "combined_score": _combined_score(recommendation, influence, alpha=alpha),
        "history": model.artifacts.history,
    }


def _evaluate_two_stage_model(
    model: TwoStageBaseline,
    dataset: SparseSyntheticDataset,
    graph: dict[str, torch.Tensor],
    recommendation_k: tuple[int, ...],
    alpha: float,
    device: torch.device,
) -> dict[str, Any]:
    score_matrix = model.predict_score_matrix(graph)
    recommendation = _recommendation_metrics(
        score_matrix=score_matrix,
        train_matrix=dataset.train_matrix,
        target_matrix=dataset.test_matrix,
        k_values=recommendation_k,
    )
    with torch.no_grad():
        user_repr, item_repr = model.recommender.encode(graph)
    seed_lookup = {
        int(item_idx): model.select_seeds(int(item_idx), user_repr=user_repr, item_repr=item_repr)
        for item_idx in dataset.evaluation_items
    }
    influence, per_item = _evaluate_influence_model(dataset, graph, seed_lookup, device=device)
    return {
        "recommendation": recommendation,
        "influence": influence,
        "per_item": per_item,
        "seeds": seed_lookup,
        "combined_score": _combined_score(recommendation, influence, alpha=alpha),
        "history": model.artifacts.history,
    }


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_torch_experiment(config: TorchExperimentConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.recsys.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = SparseSyntheticDatasetGenerator(config.dataset).generate()
    dataset.save(output_dir)
    graph = dataset.to_torch(device)

    recsys_model = _train_model(TorchRecSysOnly(dataset, config.recsys), dataset, graph, config.recsys, output_dir)
    jrso_model = _train_model(TorchJRSO(dataset, config.jrso), dataset, graph, config.jrso, output_dir)
    jrso_second_model = _train_model(
        TorchJRSOSecond(dataset, config.jrso_second),
        dataset,
        graph,
        config.jrso_second,
        output_dir,
    )
    jrso_third_model = _train_model(
        TorchJRSOThird(dataset, config.jrso_third),
        dataset,
        graph,
        config.jrso_third,
        output_dir,
    )

    im_only_model = InfluenceOnlyBaseline(dataset)
    two_stage_model = TwoStageBaseline(dataset, recsys_model, graph)

    results = {
        recsys_model.model_name: _evaluate_trainable_model(
            recsys_model,
            dataset,
            graph,
            config.recommendation_k,
            alpha=config.combined_alpha,
            device=device,
        ),
        im_only_model.model_name: _evaluate_seed_only_model(
            im_only_model,
            dataset,
            graph,
            config.recommendation_k,
            alpha=config.combined_alpha,
            device=device,
        ),
        two_stage_model.model_name: _evaluate_two_stage_model(
            two_stage_model,
            dataset,
            graph,
            config.recommendation_k,
            alpha=config.combined_alpha,
            device=device,
        ),
        jrso_model.model_name: _evaluate_trainable_model(
            jrso_model,
            dataset,
            graph,
            config.recommendation_k,
            alpha=config.combined_alpha,
            device=device,
        ),
        jrso_second_model.model_name: _evaluate_trainable_model(
            jrso_second_model,
            dataset,
            graph,
            config.recommendation_k,
            alpha=config.combined_alpha,
            device=device,
        ),
        jrso_third_model.model_name: _evaluate_trainable_model(
            jrso_third_model,
            dataset,
            graph,
            config.recommendation_k,
            alpha=config.combined_alpha,
            device=device,
        ),
    }

    history_payload = {
        recsys_model.model_name: recsys_model.artifacts.history,
        jrso_model.model_name: jrso_model.artifacts.history,
        jrso_second_model.model_name: jrso_second_model.artifacts.history,
        jrso_third_model.model_name: jrso_third_model.artifacts.history,
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
        "JRSO_third_vs_RecSys-Only_weighted_spread": paired_permutation_test(
            results["JRSO_third"]["per_item"]["weighted_spread_mean"],
            results["RecSys-Only"]["per_item"]["weighted_spread_mean"],
            seed=config.seed + 11,
        ),
        "JRSO_third_vs_Two-Stage_weighted_spread": paired_permutation_test(
            results["JRSO_third"]["per_item"]["weighted_spread_mean"],
            results["Two-Stage"]["per_item"]["weighted_spread_mean"],
            seed=config.seed + 17,
        ),
        "JRSO_vs_RecSys-Only_weighted_spread": paired_permutation_test(
            results["JRSO"]["per_item"]["weighted_spread_mean"],
            results["RecSys-Only"]["per_item"]["weighted_spread_mean"],
            seed=config.seed + 23,
        ),
        "JRSO_second_vs_Two-Stage_weighted_spread": paired_permutation_test(
            results["JRSO_second"]["per_item"]["weighted_spread_mean"],
            results["Two-Stage"]["per_item"]["weighted_spread_mean"],
            seed=config.seed + 29,
        ),
    }
    uplift = _summarize_uplift(results)

    (output_dir / "torch_training_logs.json").write_text(
        json.dumps(history_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "torch_results.json").write_text(
        json.dumps({"config": config.to_dict(), "results": compact_results, "uplift": uplift}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "torch_significance.json").write_text(
        json.dumps(significance, indent=2),
        encoding="utf-8",
    )
    (output_dir / "torch_seed_summary.json").write_text(
        json.dumps({name: payload["seeds"] for name, payload in results.items()}, indent=2),
        encoding="utf-8",
    )
    _write_metrics_csv(results, output_dir / "torch_metrics.csv")

    save_training_curve_svg(history_payload, output_dir / "torch_training_curves.svg")
    save_metric_bar_chart(compact_results, "ndcg@10", output_dir / "torch_recommendation_comparison.svg")
    save_metric_bar_chart(compact_results, "weighted_spread_rate", output_dir / "torch_influence_comparison.svg")
    save_pareto_svg(pareto_points, frontier, output_dir / "torch_pareto_frontier.svg")

    return {
        "backend": "torch",
        "device": str(device),
        "config": config.to_dict(),
        "results": compact_results,
        "uplift": uplift,
        "pareto_frontier": frontier,
        "significance": significance,
        "output_dir": str(output_dir),
    }
