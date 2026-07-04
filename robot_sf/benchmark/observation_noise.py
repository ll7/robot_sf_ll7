"""Configurable observation-noise injection for benchmark planner inputs."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

NO_OBSERVATION_NOISE_PROFILE = "none"
OBSERVATION_NOISE_INTERPRETATION = (
    "non_calibrated_benchmark_robustness_noise_not_a_real_sensor_model"
)

_DEFAULT_SPEC: dict[str, Any] = {
    "enabled": False,
    "profile": NO_OBSERVATION_NOISE_PROFILE,
    "seed": None,
    "pose_noise_std_m": 0.0,
    "heading_noise_std_rad": 0.0,
    "lidar_dropout_prob": 0.0,
    "lidar_dropout_value": 0.0,
    "pedestrian_position_noise_std_m": 0.0,
    "pedestrian_false_negative_prob": 0.0,
    "pedestrian_occlusion_max_range_m": None,
    "observation_delay_steps": 0,
    "pedestrian_false_positive_prob": 0.0,
    "pedestrian_false_positive_radius_m": 4.0,
    "pedestrian_false_positive_radius": 0.35,
    "interpretation": OBSERVATION_NOISE_INTERPRETATION,
}

_LIDAR_KEYS = {"lidar", "lidar_rays", "laser", "laser_scan", "range", "ranges", "rays"}


@dataclass
class ObservationNoiseState:
    """Mutable per-episode state for temporal planner-input perturbations."""

    delay_steps: int = 0
    pedestrian_delay_buffer: deque[dict[str, Any]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        """Normalize delay-buffer capacity."""

        if self.delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        capacity = max(1, self.delay_steps + 1)
        if self.pedestrian_delay_buffer.maxlen != capacity:
            self.pedestrian_delay_buffer = deque(
                self.pedestrian_delay_buffer,
                maxlen=capacity,
            )


def observation_noise_hash(spec: dict[str, Any]) -> str:
    """Return a stable short hash for a normalized observation-noise spec.

    Returns:
        Twelve-character SHA-1 prefix for provenance fields.
    """

    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def normalize_observation_noise_spec(spec: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize optional observation-noise configuration.

    Returns:
        Canonical JSON-serializable observation-noise specification.
    """

    if spec is None:
        return dict(_DEFAULT_SPEC)
    if not isinstance(spec, dict):
        raise TypeError("observation_noise must be a mapping")

    normalized = dict(_DEFAULT_SPEC)
    normalized.update(spec)

    enabled_explicit = "enabled" in spec
    _normalize_float_fields(normalized)

    delay_steps = int(normalized.get("observation_delay_steps", 0) or 0)
    if delay_steps < 0:
        raise ValueError("observation_delay_steps must be >= 0")
    normalized["observation_delay_steps"] = delay_steps

    normalized["pedestrian_occlusion_max_range_m"] = _normalize_optional_positive_float(
        normalized.get("pedestrian_occlusion_max_range_m"),
        "pedestrian_occlusion_max_range_m",
    )

    normalized["profile"] = str(normalized.get("profile") or NO_OBSERVATION_NOISE_PROFILE)
    normalized["interpretation"] = str(
        normalized.get("interpretation") or OBSERVATION_NOISE_INTERPRETATION
    )
    seed = normalized.get("seed")
    normalized["seed"] = int(seed) if seed is not None else None

    active = (
        any(
            float(normalized[key]) > 0.0
            for key in (
                "pose_noise_std_m",
                "heading_noise_std_rad",
                "lidar_dropout_prob",
                "pedestrian_position_noise_std_m",
                "pedestrian_false_negative_prob",
                "pedestrian_false_positive_prob",
                "observation_delay_steps",
            )
        )
        or normalized["pedestrian_occlusion_max_range_m"] is not None
    )
    normalized["enabled"] = bool(normalized.get("enabled", False)) if enabled_explicit else active
    if not normalized["enabled"]:
        normalized["profile"] = NO_OBSERVATION_NOISE_PROFILE
    return normalized


def _normalize_float_fields(normalized: dict[str, Any]) -> None:
    """Normalize scalar float fields in place."""

    for key in (
        "pose_noise_std_m",
        "heading_noise_std_rad",
        "lidar_dropout_prob",
        "lidar_dropout_value",
        "pedestrian_position_noise_std_m",
        "pedestrian_false_negative_prob",
        "pedestrian_false_positive_prob",
        "pedestrian_false_positive_radius_m",
        "pedestrian_false_positive_radius",
    ):
        normalized[key] = float(normalized.get(key, 0.0) or 0.0)
        if key.endswith("_prob") and not 0.0 <= normalized[key] <= 1.0:
            raise ValueError(f"{key} must be in [0, 1]")
        if (
            key.endswith("_std_m") or key.endswith("_std_rad") or key.endswith("_radius_m")
        ) and normalized[key] < 0.0:
            raise ValueError(f"{key} must be >= 0")


def _normalize_optional_positive_float(value: Any, key: str) -> float | None:
    """Normalize an optional positive float.

    Returns:
        Positive float when provided; otherwise ``None``.
    """

    if value is None:
        return None
    normalized = float(value)
    if normalized <= 0.0:
        raise ValueError(f"{key} must be positive when set")
    return normalized


def load_observation_noise_spec(path: str | Path) -> dict[str, Any]:
    """Load and normalize an observation-noise YAML file.

    Returns:
        Canonical observation-noise specification.
    """

    noise_path = Path(path)
    data = yaml.safe_load(noise_path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict) and isinstance(data.get("observation_noise"), dict):
        data = data["observation_noise"]
    return normalize_observation_noise_spec(data)


def make_observation_noise_rng(
    spec: dict[str, Any],
    *,
    seed: int,
    scenario_id: str,
) -> np.random.Generator:
    """Create a deterministic per-episode RNG for observation corruption.

    Returns:
        NumPy generator scoped to the scenario, episode seed, and profile.
    """

    spec_seed = spec.get("seed")
    material = {
        "episode_seed": int(seed),
        "profile": spec.get("profile", NO_OBSERVATION_NOISE_PROFILE),
        "scenario_id": str(scenario_id),
        "spec_hash": observation_noise_hash(spec),
        "spec_seed": int(spec_seed) if spec_seed is not None else None,
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))


def make_observation_noise_state(spec: dict[str, Any]) -> ObservationNoiseState:
    """Create per-episode mutable state for observation-noise application.

    Returns:
        Mutable state object used across steps in one episode.
    """

    return ObservationNoiseState(delay_steps=int(spec.get("observation_delay_steps", 0) or 0))


def new_observation_noise_stats() -> dict[str, int]:
    """Return zeroed per-episode observation-noise counters.

    Returns:
        Counter mapping for per-step or per-episode observation-noise stats.
    """

    return {
        "steps_with_noise": 0,
        "pose_noise_applied": 0,
        "heading_noise_applied": 0,
        "lidar_values_dropped": 0,
        "pedestrian_position_noise_applied": 0,
        "pedestrians_removed": 0,
        "pedestrians_occluded": 0,
        "observation_delay_applied": 0,
        "pedestrians_added": 0,
    }


def merge_observation_noise_stats(total: dict[str, int], step: dict[str, int]) -> None:
    """Accumulate step-level observation-noise counters into an episode total."""

    for key, value in step.items():
        total[key] = int(total.get(key, 0)) + int(value)


def apply_observation_noise(
    obs: dict[str, Any],
    spec: dict[str, Any],
    rng: np.random.Generator,
    state: ObservationNoiseState | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Apply benchmark observation noise to one planner input observation.

    Returns:
        Pair of noisy observation copy and step-level observation-noise counters.
    """

    stats = new_observation_noise_stats()
    if not bool(spec.get("enabled", False)):
        return obs, stats
    noisy = deepcopy(obs)
    _apply_pose_noise(noisy, spec, rng, stats)
    _apply_lidar_dropout(noisy, spec, rng, stats)
    _apply_pedestrian_noise(noisy, spec, rng, stats)
    _apply_observation_delay(noisy, spec, state, stats)
    if any(value > 0 for key, value in stats.items() if key != "steps_with_noise"):
        stats["steps_with_noise"] = 1
    return noisy, stats


def _as_xy_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size < 2:
        return None
    return arr.reshape(-1)[:2].astype(float, copy=True)


def _robot_position(obs: dict[str, Any]) -> np.ndarray:
    robot_pos = _as_xy_array(obs.get("robot_position"))
    if robot_pos is not None:
        return robot_pos
    robot = obs.get("robot") if isinstance(obs.get("robot"), dict) else {}
    robot_pos = _as_xy_array(robot.get("position"))
    if robot_pos is not None:
        return robot_pos
    return np.zeros(2, dtype=float)


def _apply_pose_noise(
    obs: dict[str, Any],
    spec: dict[str, Any],
    rng: np.random.Generator,
    stats: dict[str, int],
) -> None:
    pose_std = float(spec.get("pose_noise_std_m", 0.0))
    heading_std = float(spec.get("heading_noise_std_rad", 0.0))
    robot = obs.get("robot") if isinstance(obs.get("robot"), dict) else None
    if pose_std > 0.0:
        delta = rng.normal(0.0, pose_std, size=2)
        pose_mutated = False
        if robot is not None and "position" in robot:
            base = _as_xy_array(robot["position"])
            if base is not None:
                robot["position"] = (base + delta).tolist()
                pose_mutated = True
        if "robot_position" in obs:
            base = _as_xy_array(obs["robot_position"])
            if base is not None:
                obs["robot_position"] = (base + delta).tolist()
                pose_mutated = True
        if pose_mutated:
            stats["pose_noise_applied"] += 1
    if heading_std <= 0.0:
        return
    delta_heading = float(rng.normal(0.0, heading_std))
    if robot is not None and "heading" in robot:
        robot["heading"] = _with_heading_delta(robot["heading"], delta_heading)
        stats["heading_noise_applied"] += 1
    if "robot_heading" in obs:
        obs["robot_heading"] = _with_heading_delta(obs["robot_heading"], delta_heading)
        stats["heading_noise_applied"] += 1


def _with_heading_delta(value: Any, delta_heading: float) -> float | list[float]:
    heading = np.asarray(value, dtype=float)
    if heading.shape == ():
        return float(heading + delta_heading)
    adjusted = heading.astype(float, copy=True)
    adjusted.flat[0] += delta_heading
    return adjusted.tolist()


def _apply_lidar_dropout(
    obs: dict[str, Any],
    spec: dict[str, Any],
    rng: np.random.Generator,
    stats: dict[str, int],
) -> None:
    dropout_prob = float(spec.get("lidar_dropout_prob", 0.0))
    if dropout_prob <= 0.0:
        return
    dropout_value = float(spec.get("lidar_dropout_value", 0.0))

    def _visit(node: Any) -> Any:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                key_norm = str(key).strip().lower()
                if key_norm in _LIDAR_KEYS:
                    node[key] = _drop_numeric_values(value, dropout_prob, dropout_value, rng, stats)
                elif isinstance(value, dict):
                    _visit(value)
        return node

    _visit(obs)


def _drop_numeric_values(
    value: Any,
    dropout_prob: float,
    dropout_value: float,
    rng: np.random.Generator,
    stats: dict[str, int],
) -> Any:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return value
    if arr.size == 0:
        return value
    dropped = rng.random(arr.shape) < dropout_prob
    out = arr.astype(float, copy=True)
    out[dropped] = dropout_value
    stats["lidar_values_dropped"] += int(np.count_nonzero(dropped))
    return out.tolist()


def _apply_pedestrian_noise(
    obs: dict[str, Any],
    spec: dict[str, Any],
    rng: np.random.Generator,
    stats: dict[str, int],
) -> None:
    pedestrians = obs.get("pedestrians")
    if not isinstance(pedestrians, dict):
        _apply_flat_pedestrian_noise(obs, spec, rng, stats)
        return

    positions = np.asarray(pedestrians.get("positions", []), dtype=float).reshape(-1, 2)
    velocities = np.asarray(pedestrians.get("velocities", []), dtype=float).reshape(-1, 2)
    radii = np.asarray(pedestrians.get("radius", []), dtype=float).reshape(-1)
    if velocities.shape[0] != positions.shape[0]:
        velocities = np.zeros_like(positions)
    if radii.shape[0] != positions.shape[0]:
        radii = np.full((positions.shape[0],), 0.35, dtype=float)

    fn_prob = float(spec.get("pedestrian_false_negative_prob", 0.0))
    if fn_prob > 0.0 and positions.shape[0] > 0:
        keep = rng.random(positions.shape[0]) >= fn_prob
        removed = int(positions.shape[0] - np.count_nonzero(keep))
        positions = positions[keep]
        velocities = velocities[keep]
        radii = radii[keep]
        stats["pedestrians_removed"] += removed

    positions, velocities, radii = _apply_range_occlusion(
        obs,
        positions,
        velocities,
        radii,
        spec,
        stats,
    )
    positions = _apply_pedestrian_position_noise(positions, spec, rng, stats)

    fp_prob = float(spec.get("pedestrian_false_positive_prob", 0.0))
    if fp_prob > 0.0 and float(rng.random()) < fp_prob:
        robot_pos = _robot_position(obs)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius_m = float(rng.uniform(0.0, spec.get("pedestrian_false_positive_radius_m", 4.0)))
        fp_pos = robot_pos + np.array([np.cos(angle), np.sin(angle)], dtype=float) * radius_m
        positions = np.vstack([positions, fp_pos.reshape(1, 2)])
        velocities = np.vstack([velocities, np.zeros((1, 2), dtype=float)])
        radii = np.concatenate(
            [
                radii,
                np.asarray([float(spec.get("pedestrian_false_positive_radius", 0.35))]),
            ]
        )
        stats["pedestrians_added"] += 1

    pedestrians["positions"] = positions.tolist()
    pedestrians["velocities"] = velocities.tolist()
    pedestrians["radius"] = radii.tolist()
    pedestrians["count"] = int(positions.shape[0])


def _apply_flat_pedestrian_noise(
    obs: dict[str, Any],
    spec: dict[str, Any],
    rng: np.random.Generator,
    stats: dict[str, int],
) -> None:
    if "pedestrians_positions" not in obs:
        return
    positions_buf = np.asarray(obs.get("pedestrians_positions", []), dtype=float).reshape(-1, 2)
    if positions_buf.size == 0:
        return
    velocities_buf = np.asarray(obs.get("pedestrians_velocities", []), dtype=float).reshape(-1, 2)
    if velocities_buf.shape[0] != positions_buf.shape[0]:
        velocities_buf = np.zeros_like(positions_buf)
    count_arr = np.asarray(
        obs.get("pedestrians_count", [positions_buf.shape[0]]), dtype=float
    ).reshape(-1)
    active_count = int(count_arr[0]) if count_arr.size else int(positions_buf.shape[0])
    active_count = max(0, min(active_count, positions_buf.shape[0]))
    positions = positions_buf[:active_count].copy()
    velocities = velocities_buf[:active_count].copy()

    fn_prob = float(spec.get("pedestrian_false_negative_prob", 0.0))
    if fn_prob > 0.0 and positions.shape[0] > 0:
        keep = rng.random(positions.shape[0]) >= fn_prob
        removed = int(positions.shape[0] - np.count_nonzero(keep))
        positions = positions[keep]
        velocities = velocities[keep]
        stats["pedestrians_removed"] += removed

    positions, velocities, _ = _apply_range_occlusion(
        obs,
        positions,
        velocities,
        np.full((positions.shape[0],), 0.35, dtype=float),
        spec,
        stats,
    )
    positions = _apply_pedestrian_position_noise(positions, spec, rng, stats)

    fp_prob = float(spec.get("pedestrian_false_positive_prob", 0.0))
    if (
        fp_prob > 0.0
        and positions.shape[0] < positions_buf.shape[0]
        and float(rng.random()) < fp_prob
    ):
        robot_pos = _robot_position(obs)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius_m = float(rng.uniform(0.0, spec.get("pedestrian_false_positive_radius_m", 4.0)))
        fp_pos = robot_pos + np.array([np.cos(angle), np.sin(angle)], dtype=float) * radius_m
        positions = np.vstack([positions, fp_pos.reshape(1, 2)])
        velocities = np.vstack([velocities, np.zeros((1, 2), dtype=float)])
        stats["pedestrians_added"] += 1

    out_positions = np.array(positions_buf, dtype=float, copy=True)
    out_velocities = np.array(velocities_buf, dtype=float, copy=True)
    out_positions[:] = 0.0
    out_velocities[:] = 0.0
    out_positions[: positions.shape[0]] = positions
    out_velocities[: velocities.shape[0]] = velocities
    obs["pedestrians_positions"] = out_positions.tolist()
    obs["pedestrians_velocities"] = out_velocities.tolist()
    obs["pedestrians_count"] = [float(positions.shape[0])]


def _apply_range_occlusion(
    obs: dict[str, Any],
    positions: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    spec: dict[str, Any],
    stats: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    range_limit = spec.get("pedestrian_occlusion_max_range_m")
    if range_limit is None or positions.shape[0] == 0:
        return positions, velocities, radii
    robot_pos = _robot_position(obs)
    distances = np.linalg.norm(positions - robot_pos.reshape(1, 2), axis=1)
    keep = distances <= float(range_limit)
    occluded = int(positions.shape[0] - np.count_nonzero(keep))
    if occluded <= 0:
        return positions, velocities, radii
    stats["pedestrians_occluded"] += occluded
    return positions[keep], velocities[keep], radii[keep]


def _apply_pedestrian_position_noise(
    positions: np.ndarray,
    spec: dict[str, Any],
    rng: np.random.Generator,
    stats: dict[str, int],
) -> np.ndarray:
    """Apply Gaussian noise to planner-facing pedestrian coordinates.

    Returns:
        Pedestrian positions with deterministic Gaussian offsets applied.
    """

    noise_std = float(spec.get("pedestrian_position_noise_std_m", 0.0))
    if noise_std <= 0.0 or positions.shape[0] == 0:
        return positions
    stats["pedestrian_position_noise_applied"] += int(positions.shape[0])
    return positions + rng.normal(0.0, noise_std, size=positions.shape)


def _pedestrian_snapshot(obs: dict[str, Any]) -> dict[str, Any] | None:
    pedestrians = obs.get("pedestrians")
    if isinstance(pedestrians, dict):
        return {"kind": "structured", "pedestrians": deepcopy(pedestrians)}
    if "pedestrians_positions" in obs:
        return {
            "kind": "flat",
            "pedestrians_positions": deepcopy(obs.get("pedestrians_positions")),
            "pedestrians_velocities": deepcopy(obs.get("pedestrians_velocities")),
            "pedestrians_count": deepcopy(obs.get("pedestrians_count")),
        }
    return None


def _restore_pedestrian_snapshot(obs: dict[str, Any], snapshot: dict[str, Any]) -> None:
    if snapshot.get("kind") == "structured":
        obs["pedestrians"] = deepcopy(snapshot["pedestrians"])
        return
    obs["pedestrians_positions"] = deepcopy(snapshot.get("pedestrians_positions"))
    obs["pedestrians_velocities"] = deepcopy(snapshot.get("pedestrians_velocities"))
    obs["pedestrians_count"] = deepcopy(snapshot.get("pedestrians_count"))


def _apply_observation_delay(
    obs: dict[str, Any],
    spec: dict[str, Any],
    state: ObservationNoiseState | None,
    stats: dict[str, int],
) -> None:
    delay_steps = int(spec.get("observation_delay_steps", 0) or 0)
    if delay_steps <= 0:
        return
    if state is None:
        # Fail closed: observation delay needs persistent per-episode state to
        # carry pedestrian snapshots across steps. A transient state created
        # here would be discarded every call, silently yielding a no-op delay
        # (under-reported perception degradation). Callers must thread the
        # state from ``make_observation_noise_state``.
        raise ValueError(
            "observation_delay_steps > 0 requires a persistent ObservationNoiseState; "
            "pass the state from make_observation_noise_state()"
        )
    snapshot = _pedestrian_snapshot(obs)
    if snapshot is None:
        return
    state.pedestrian_delay_buffer.append(snapshot)
    if len(state.pedestrian_delay_buffer) <= delay_steps:
        return
    delayed_snapshot = state.pedestrian_delay_buffer[0]
    _restore_pedestrian_snapshot(obs, delayed_snapshot)
    stats["observation_delay_applied"] += 1
