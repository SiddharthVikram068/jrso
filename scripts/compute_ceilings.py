"""Compute the theoretical and practical ceilings for every metric reported by the
Ciao backend. Loads the saved ``ciao_dataset.npz`` so we use exactly the same
splits, edge probabilities, and score-matrix proxy that the experiment used.

Run:
    python scripts/compute_ceilings.py outputs/ciao_medium_real
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def main(run_dir: Path) -> None:
    npz = np.load(run_dir / "ciao_dataset.npz")
    train_matrix = npz["train_matrix"].astype(bool)
    test_matrix = npz["test_matrix"].astype(bool)
    score_matrix = npz["score_matrix"].astype(np.float32)
    edge_index = npz["edge_index"]
    edge_weight = npz["edge_weight"].astype(np.float32)
    eval_items = npz["evaluation_items"].astype(np.int64)
    num_users, num_items = train_matrix.shape

    meta = json.loads((run_dir / "ciao_dataset_meta.json").read_text(encoding="utf-8"))
    config = meta["config"]
    k_seeds = int(config["k_seeds"])

    print("Dataset shape:", train_matrix.shape, "edges:", edge_index.shape[1])
    print("k_seeds =", k_seeds, "  num_eval_items =", eval_items.size)
    print()

    print("=" * 64)
    print("RECOMMENDATION CEILINGS  (per user, averaged over eligible test users)")
    print("=" * 64)
    eligible = np.flatnonzero(test_matrix.sum(axis=1) > 0)
    truth_sizes = test_matrix.sum(axis=1)[eligible]
    for k in (10, 20):
        max_recall_k = 1.0
        max_precision_k = float(np.mean(np.clip(truth_sizes, 0, k) / k))
        max_ndcg_k = 1.0
        print(
            f"k={k:>2}: max recall = {max_recall_k:6.4f}   "
            f"max precision = {max_precision_k:6.4f}   "
            f"max NDCG = {max_ndcg_k:6.4f}"
        )
    print(
        f"  (max precision@k is capped by min(|truth|, k) / k; "
        f"avg test positives per user = {float(truth_sizes.mean()):.2f})"
    )

    print()
    print("=" * 64)
    print("INFLUENCE CEILINGS  (averaged over evaluation items)")
    print("=" * 64)
    print(f"max spread_rate      = 1.0000  (every user activated)")

    eval_scores = score_matrix[:, eval_items]
    item_means = eval_scores.mean(axis=0)
    print(
        f"max weighted_spread_rate (all users active)        = {float(item_means.mean()):.4f}"
    )
    topk_per_item = np.sort(eval_scores, axis=0)[-k_seeds:].mean(axis=0)
    print(
        f"  (note: weighted_spread_rate is bounded by the mean of the score column;\n"
        f"   for perfectly-targeted top-{k_seeds} seed users the mean score among seeds is "
        f"{float(topk_per_item.mean()):.4f})"
    )

    print(
        f"max test_positive_recall = 1.0000  "
        f"(simulator activates every test-positive user)"
    )

    print()
    print("=" * 64)
    print("COMBINED SCORE CEILING  ( alpha*NDCG@10 + (1-alpha)*weighted_spread_rate )")
    print("=" * 64)
    ws_ceiling = float(item_means.mean())
    for alpha in (0.5, 0.7, 0.3):
        ceiling = alpha * 1.0 + (1.0 - alpha) * ws_ceiling
        print(
            f"alpha={alpha:.1f}: theoretical max combined = {ceiling:.4f}   "
            f"(NDCG=1.0, wSpread={ws_ceiling:.4f})"
        )

    print()
    print("=" * 64)
    print("PRACTICAL ORACLES  (what an unconstrained, perfectly-tuned model could hit)")
    print("=" * 64)
    monte_carlo_seeds = min(k_seeds, num_users)
    practical_ndcg_ceiling = 0.30
    fully_active_ws = float(item_means.mean())
    practical_combined_at_05 = 0.5 * practical_ndcg_ceiling + 0.5 * fully_active_ws
    print(
        f"Realistic NDCG@10 ceiling on Ciao (LightGCN/SBPR SOTA reports ~0.05 to ~0.10) "
        f"  ~= 0.10"
    )
    print(
        f"Realistic weighted_spread_rate ceiling                 ~= "
        f"{fully_active_ws:.4f}  (full saturation)"
    )
    print(
        f"Realistic alpha=0.5 combined ceiling                   ~= "
        f"{0.5 * 0.10 + 0.5 * fully_active_ws:.4f}"
    )

    print()
    print("Current observed values for the JRSO_second model on this dataset:")
    summary = json.loads((run_dir / "ciao_results.json").read_text(encoding="utf-8"))
    if "JRSO_second" in summary["results"]:
        payload = summary["results"]["JRSO_second"]
        print(f"  NDCG@10              = {payload['ndcg@10']:.4f}")
        print(f"  weighted_spread_rate = {payload['weighted_spread_rate']:.4f}")
        print(f"  combined             = {payload['combined_score']:.4f}")
    if "JRSO" in summary["results"]:
        payload = summary["results"]["JRSO"]
        print(f"  JRSO combined        = {payload['combined_score']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        run = Path("outputs/ciao_medium_real")
    main(run)
