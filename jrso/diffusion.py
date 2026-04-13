from __future__ import annotations

from heapq import heappop, heappush
from typing import Iterable

import autograd.numpy as anp
import numpy as np


def sigmoid(x: np.ndarray | anp.ndarray) -> np.ndarray | anp.ndarray:
    return 1.0 / (1.0 + anp.exp(-x))


def softmax(x: np.ndarray | anp.ndarray) -> np.ndarray | anp.ndarray:
    shifted = x - anp.max(x)
    exp_x = anp.exp(shifted)
    return exp_x / (anp.sum(exp_x) + 1e-8)


def logit(p: np.ndarray | anp.ndarray) -> np.ndarray | anp.ndarray:
    clipped = anp.clip(p, 1e-6, 1.0 - 1e-6)
    return anp.log(clipped) - anp.log1p(-clipped)


def adoption_probability(
    base_preference: float,
    influence_sum: float,
    influence_scale: float,
) -> float:
    probability = 1.0 / (1.0 + np.exp(-(float(logit(base_preference)) + influence_scale * influence_sum)))
    return float(np.clip(probability, 0.0, 1.0))


def simulate_progressive_cascade(
    item_scores: np.ndarray,
    social_adjacency: np.ndarray,
    influence_probabilities: np.ndarray,
    seeds: Iterable[int],
    cascade_steps: int,
    influence_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    active = np.zeros(item_scores.shape[0], dtype=bool)
    seed_array = np.array(sorted(set(int(seed) for seed in seeds)), dtype=np.int64)
    active[seed_array] = True
    frontier = active.copy()
    for _ in range(cascade_steps):
        new_active = active.copy()
        frontier_nodes = np.flatnonzero(frontier)
        if frontier_nodes.size == 0:
            break
        for user_idx in np.flatnonzero(~active):
            incoming = frontier_nodes[social_adjacency[frontier_nodes, user_idx] > 0]
            if incoming.size == 0:
                continue
            influence_sum = float(influence_probabilities[incoming, user_idx].sum())
            probability = adoption_probability(item_scores[user_idx], influence_sum, influence_scale)
            if rng.random() < probability:
                new_active[user_idx] = True
        frontier = new_active & ~active
        active = new_active
    return active


def estimate_expected_spread(
    item_scores: np.ndarray,
    social_adjacency: np.ndarray,
    influence_probabilities: np.ndarray,
    seeds: Iterable[int],
    cascade_steps: int,
    influence_scale: float,
    num_trials: int,
    rng_seed: int,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    adoption_counts = np.zeros(item_scores.shape[0], dtype=np.float64)
    spread_values: list[float] = []
    for _ in range(num_trials):
        active = simulate_progressive_cascade(
            item_scores=item_scores,
            social_adjacency=social_adjacency,
            influence_probabilities=influence_probabilities,
            seeds=seeds,
            cascade_steps=cascade_steps,
            influence_scale=influence_scale,
            rng=rng,
        )
        adoption_counts += active.astype(np.float64)
        spread_values.append(float(active.sum()))
    return float(np.mean(spread_values)), adoption_counts / max(num_trials, 1)


def celf_select_seeds(
    item_scores: np.ndarray,
    social_adjacency: np.ndarray,
    influence_probabilities: np.ndarray,
    k: int,
    cascade_steps: int,
    influence_scale: float,
    num_trials: int,
    rng_seed: int,
) -> list[int]:
    def spread(seed_set: list[int], offset: int) -> float:
        value, _ = estimate_expected_spread(
            item_scores=item_scores,
            social_adjacency=social_adjacency,
            influence_probabilities=influence_probabilities,
            seeds=seed_set,
            cascade_steps=cascade_steps,
            influence_scale=influence_scale,
            num_trials=num_trials,
            rng_seed=rng_seed + offset,
        )
        return value

    seeds: list[int] = []
    heap: list[tuple[float, int, int]] = []
    num_users = item_scores.shape[0]
    for user_idx in range(num_users):
        gain = spread([user_idx], user_idx)
        heappush(heap, (-gain, user_idx, 0))

    current_spread = 0.0
    while len(seeds) < min(k, num_users):
        neg_gain, user_idx, last_updated = heappop(heap)
        if last_updated == len(seeds):
            seeds.append(user_idx)
            current_spread += -neg_gain
            continue
        new_spread = spread(seeds + [user_idx], len(seeds) * 10_000 + user_idx)
        marginal_gain = new_spread - spread(seeds, len(seeds) * 5_000 + user_idx)
        heappush(heap, (-marginal_gain, user_idx, len(seeds)))
    return seeds


def continuous_seed_probabilities(logits: anp.ndarray, budget: int) -> anp.ndarray:
    weights = softmax(logits)
    return 1.0 - anp.exp(-budget * weights)


def differentiable_diffusion(
    user_embeddings: anp.ndarray,
    item_embedding: anp.ndarray,
    social_adjacency: np.ndarray,
    seed_bias: anp.ndarray,
    k_seeds: int,
    cascade_steps: int,
) -> tuple[anp.ndarray, anp.ndarray, anp.ndarray]:
    dim = user_embeddings.shape[1]
    affinity = sigmoid((user_embeddings @ item_embedding) / anp.sqrt(dim))
    social_logits = (user_embeddings @ user_embeddings.T) / anp.sqrt(dim)
    source_strength = affinity[:, None]
    target_receptivity = affinity[None, :]
    edge_influence = social_adjacency * sigmoid(social_logits) * source_strength * target_receptivity
    edge_influence = anp.clip(edge_influence, 0.0, 0.95)

    seed_logits = affinity + seed_bias
    activations = continuous_seed_probabilities(seed_logits, k_seeds)
    for _ in range(cascade_steps):
        safe_term = anp.clip(1.0 - edge_influence * activations[:, None], 1e-6, 1.0)
        prod_term = anp.exp(anp.sum(anp.log(safe_term), axis=0))
        activations = 1.0 - (1.0 - activations) * prod_term
    expected_spread = anp.mean(activations)
    return expected_spread, activations, affinity
