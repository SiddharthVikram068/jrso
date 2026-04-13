from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdamState:
    first_moment: dict[str, np.ndarray]
    second_moment: dict[str, np.ndarray]
    step: int = 0


def initialize_adam(parameters: dict[str, np.ndarray]) -> AdamState:
    return AdamState(
        first_moment={name: np.zeros_like(value) for name, value in parameters.items()},
        second_moment={name: np.zeros_like(value) for name, value in parameters.items()},
    )


def adam_step(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    state: AdamState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[dict[str, np.ndarray], AdamState]:
    state.step += 1
    updated: dict[str, np.ndarray] = {}
    for name, value in parameters.items():
        grad = np.asarray(gradients[name], dtype=np.float64)
        state.first_moment[name] = beta1 * state.first_moment[name] + (1.0 - beta1) * grad
        state.second_moment[name] = beta2 * state.second_moment[name] + (1.0 - beta2) * (grad**2)
        m_hat = state.first_moment[name] / (1.0 - beta1**state.step)
        v_hat = state.second_moment[name] / (1.0 - beta2**state.step)
        updated[name] = np.asarray(
            value - learning_rate * m_hat / (np.sqrt(v_hat) + eps),
            dtype=np.float64,
        )
    return updated, state
