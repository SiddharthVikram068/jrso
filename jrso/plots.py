from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _write_svg(path: Path, body: str, width: int = 720, height: int = 420) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="#faf8f4"/>'
        f"{body}</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def save_training_curve_svg(history_by_model: dict[str, list[dict[str, float]]], path: Path) -> None:
    width, height = 720, 420
    margin = 60
    all_values = [point["objective"] for history in history_by_model.values() for point in history]
    if not all_values:
        _write_svg(path, "", width=width, height=height)
        return
    y_min, y_max = min(all_values), max(all_values)
    y_span = max(y_max - y_min, 1e-6)
    colors = ["#c0532e", "#235789", "#48a868", "#8a5cff", "#1c1c1c"]
    parts = [
        f'<text x="{margin}" y="32" font-size="22" font-family="Georgia" fill="#1f1f1f">Training Objective</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#444" />',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#444" />',
    ]
    max_epoch = max(len(history) for history in history_by_model.values()) - 1 or 1
    for index, (name, history) in enumerate(history_by_model.items()):
        color = colors[index % len(colors)]
        coords = []
        for point in history:
            epoch = point["epoch"]
            value = point["objective"]
            x = margin + (width - 2 * margin) * (epoch / max_epoch)
            y = height - margin - (height - 2 * margin) * ((value - y_min) / y_span)
            coords.append(f"{x:.1f},{y:.1f}")
        if coords:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(coords)}" />')
        parts.append(
            f'<text x="{width - margin - 140}" y="{margin + 22 * index}" font-size="14" font-family="Arial" fill="{color}">{name}</text>'
        )
    _write_svg(path, "".join(parts), width=width, height=height)


def save_metric_bar_chart(metrics_by_model: dict[str, dict[str, float]], metric_name: str, path: Path) -> None:
    width, height = 720, 420
    margin = 60
    models = list(metrics_by_model)
    values = [metrics_by_model[name].get(metric_name, 0.0) for name in models]
    max_value = max(values) if values else 1.0
    colors = ["#c0532e", "#235789", "#48a868", "#8a5cff"]
    parts = [
        f'<text x="{margin}" y="32" font-size="22" font-family="Georgia" fill="#1f1f1f">{metric_name}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#444" />',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#444" />',
    ]
    bar_width = (width - 2 * margin) / max(len(models), 1)
    for index, (name, value) in enumerate(zip(models, values)):
        left = margin + index * bar_width + 12
        bar_height = 0 if max_value == 0 else (height - 2 * margin) * (value / max_value)
        top = height - margin - bar_height
        parts.append(
            f'<rect x="{left:.1f}" y="{top:.1f}" width="{bar_width - 24:.1f}" height="{bar_height:.1f}" '
            f'fill="{colors[index % len(colors)]}" rx="6" />'
        )
        parts.append(
            f'<text x="{left:.1f}" y="{height - margin + 18}" font-size="13" font-family="Arial" fill="#1f1f1f">{name}</text>'
        )
        parts.append(
            f'<text x="{left:.1f}" y="{top - 8:.1f}" font-size="12" font-family="Arial" fill="#1f1f1f">{value:.3f}</text>'
        )
    _write_svg(path, "".join(parts), width=width, height=height)


def save_pareto_svg(points: dict[str, tuple[float, float]], frontier: dict[str, tuple[float, float]], path: Path) -> None:
    width, height = 720, 420
    margin = 60
    parts = [
        f'<text x="{margin}" y="32" font-size="22" font-family="Georgia" fill="#1f1f1f">Pareto Frontier</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#444" />',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#444" />',
    ]
    if not points:
        _write_svg(path, "".join(parts), width=width, height=height)
        return
    xs = [value[0] for value in points.values()]
    ys = [value[1] for value in points.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    frontier_points = sorted(frontier.items(), key=lambda item: item[1][0])
    frontier_coords = []
    for label, (x_value, y_value) in points.items():
        x = margin + (width - 2 * margin) * ((x_value - x_min) / x_span)
        y = height - margin - (height - 2 * margin) * ((y_value - y_min) / y_span)
        color = "#c0532e" if label in frontier else "#648381"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" />')
        parts.append(
            f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="12" font-family="Arial" fill="#1f1f1f">{label}</text>'
        )
        if label in frontier:
            frontier_coords.append((x, y))
    if frontier_coords:
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in frontier_coords)
        parts.append(f'<polyline fill="none" stroke="#c0532e" stroke-width="2.5" points="{polyline}" />')
    _write_svg(path, "".join(parts), width=width, height=height)


def save_seed_selection_svg(
    communities: np.ndarray,
    adjacency: np.ndarray,
    predicted_seeds: list[int],
    oracle_seeds: list[int],
    path: Path,
) -> None:
    width, height = 720, 520
    center_x, center_y = width / 2, height / 2 + 20
    radius = min(width, height) * 0.34
    num_users = communities.shape[0]
    community_ids = sorted(set(map(int, communities)))
    positions: dict[int, tuple[float, float]] = {}
    community_offsets = {community: idx for idx, community in enumerate(community_ids)}
    for idx in range(num_users):
        angle = 2 * math.pi * idx / max(num_users, 1)
        radial = radius - 26 * community_offsets[int(communities[idx])]
        positions[idx] = (
            center_x + radial * math.cos(angle),
            center_y + radial * math.sin(angle),
        )

    parts = [
        f'<text x="40" y="36" font-size="22" font-family="Georgia" fill="#1f1f1f">Seed Selection</text>',
    ]
    for source in range(num_users):
        for target in np.flatnonzero(adjacency[source]):
            if source < target:
                x1, y1 = positions[source]
                x2, y2 = positions[target]
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="#b7c4c2" stroke-opacity="0.32" stroke-width="1" />'
                )

    predicted = set(predicted_seeds)
    oracle = set(oracle_seeds)
    for idx, (x_coord, y_coord) in positions.items():
        fill = "#648381"
        if idx in predicted and idx in oracle:
            fill = "#48a868"
        elif idx in predicted:
            fill = "#c0532e"
        elif idx in oracle:
            fill = "#235789"
        parts.append(f'<circle cx="{x_coord:.1f}" cy="{y_coord:.1f}" r="6" fill="{fill}" />')
    _write_svg(path, "".join(parts), width=width, height=height)
