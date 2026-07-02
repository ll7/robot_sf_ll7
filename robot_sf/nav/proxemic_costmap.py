"""Analytic proxemic soft-cost layer for classical planners."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

DecayFunction = Literal["linear", "gaussian"]


@dataclass(frozen=True)
class ProxemicCostmapConfig:
    """Configuration for pedestrian personal/social-zone soft costs."""

    enabled: bool = False
    personal_radius: float = 0.45
    social_radius: float = 1.2
    personal_weight: float = 1.0
    social_weight: float = 0.35
    velocity_elongation_factor: float = 0.0
    max_cost: float = 10.0
    decay_function: DecayFunction = "linear"

    def __post_init__(self) -> None:
        """Fail closed on malformed proxemic-layer configuration."""
        _require_finite_non_negative("personal_radius", self.personal_radius)
        _require_finite_non_negative("social_radius", self.social_radius)
        if self.social_radius < self.personal_radius:
            raise ValueError("social_radius must be >= personal_radius")
        _require_finite_non_negative("personal_weight", self.personal_weight)
        _require_finite_non_negative("social_weight", self.social_weight)
        _require_finite_non_negative("velocity_elongation_factor", self.velocity_elongation_factor)
        _require_finite_non_negative("max_cost", self.max_cost)
        if self.decay_function not in {"linear", "gaussian"}:
            raise ValueError("decay_function must be one of: linear, gaussian")


def config_hash(config: ProxemicCostmapConfig | Mapping[str, Any] | None) -> str:
    """Return a stable short hash for proxemic-layer provenance metadata."""
    resolved = build_proxemic_costmap_config(config)
    payload = json.dumps(asdict(resolved), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_proxemic_costmap_config(
    config: ProxemicCostmapConfig | Mapping[str, Any] | None = None,
) -> ProxemicCostmapConfig:
    """Build a typed config from YAML-style mappings or an existing config.

    Returns:
        ProxemicCostmapConfig: Validated costmap configuration.
    """
    if config is None:
        return ProxemicCostmapConfig()
    if isinstance(config, ProxemicCostmapConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("proxemic costmap config must be a mapping or ProxemicCostmapConfig")
    allowed = set(ProxemicCostmapConfig.__dataclass_fields__)
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown proxemic costmap config fields: {unknown}")
    return ProxemicCostmapConfig(**dict(config))


def proxemic_cost_at_points(
    points_xy: np.ndarray,
    pedestrian_positions: np.ndarray,
    pedestrian_velocities: np.ndarray | None,
    config: ProxemicCostmapConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Compute per-point soft proxemic costs around pedestrians.

    The layer is analytic rather than rasterized. Velocity elongation stretches
    the personal/social zones in front of each moving pedestrian only.

    Returns:
        np.ndarray: One soft cost per query point.
    """
    cfg = build_proxemic_costmap_config(config)
    points = _as_points_array("points_xy", points_xy)
    if not cfg.enabled:
        return np.zeros(points.shape[0], dtype=float)

    ped_pos = _as_points_array("pedestrian_positions", pedestrian_positions)
    if ped_pos.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=float)

    ped_vel = _as_velocity_array(pedestrian_velocities, ped_pos.shape[0])
    costs = np.zeros(points.shape[0], dtype=float)
    for position, velocity in zip(ped_pos, ped_vel, strict=True):
        distance = _elongated_distance(points - position[None, :], velocity, cfg)
        if cfg.social_radius > 0.0 and cfg.social_weight > 0.0:
            costs += cfg.social_weight * _decay(distance, cfg.social_radius, cfg.decay_function)
        if cfg.personal_radius > 0.0 and cfg.personal_weight > 0.0:
            costs += cfg.personal_weight * _decay(distance, cfg.personal_radius, cfg.decay_function)
    return np.minimum(costs, cfg.max_cost)


def _require_finite_non_negative(name: str, value: float) -> None:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")


def _as_points_array(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_velocity_array(values: np.ndarray | None, count: int) -> np.ndarray:
    if values is None:
        return np.zeros((count, 2), dtype=float)
    arr = _as_points_array("pedestrian_velocities", values)
    if arr.shape[0] != count:
        raise ValueError("pedestrian_velocities must match pedestrian_positions length")
    return arr


def _elongated_distance(
    relative_points: np.ndarray,
    velocity: np.ndarray,
    config: ProxemicCostmapConfig,
) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-9 or config.velocity_elongation_factor <= 0.0:
        return np.linalg.norm(relative_points, axis=1)
    direction = velocity / speed
    longitudinal = relative_points @ direction
    lateral = relative_points - longitudinal[:, None] * direction[None, :]
    scale = 1.0 + config.velocity_elongation_factor * speed
    adjusted_longitudinal = np.where(longitudinal > 0.0, longitudinal / scale, longitudinal)
    return np.sqrt(adjusted_longitudinal**2 + np.sum(lateral * lateral, axis=1))


def _decay(distance: np.ndarray, radius: float, decay_function: DecayFunction) -> np.ndarray:
    if radius <= 0.0:
        return np.zeros_like(distance, dtype=float)
    if decay_function == "linear":
        return np.clip(1.0 - distance / radius, 0.0, 1.0)
    raw = np.exp(-0.5 * (distance / max(radius, 1e-12)) ** 2)
    baseline = np.exp(-0.5)
    normalized = (raw - baseline) / (1.0 - baseline)
    return np.clip(normalized, 0.0, 1.0)


__all__ = [
    "ProxemicCostmapConfig",
    "build_proxemic_costmap_config",
    "config_hash",
    "proxemic_cost_at_points",
]
