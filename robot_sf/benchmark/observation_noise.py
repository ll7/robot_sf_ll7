"""Configurable observation-noise injection for benchmark planner inputs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
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
    "pedestrian_false_negative_prob": 0.0,
    "pedestrian_false_positive_prob": 0.0,
    "pedestrian_false_positive_radius_m": 4.0,
    "pedestrian_false_positive_radius": 0.35,
    "interpretation": OBSERVATION_NOISE_INTERPRETATION,
}

_LIDAR_KEYS = {"lidar", "lidar_rays", "laser", "laser_scan", "range", "ranges", "rays"}


def observation_noise_hash(spec: dict[str, Any]) -> str:
    """Return a stable short hash for a normalized observation-noise spec."""
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def normalize_observation_noise_spec(spec: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize an optional observation-noise configuration.

    Returns:
        Canonical JSON-serializable noise specification. Omitted or zero-valued specs become a
        disabled ``profile: none`` no-op profile.
    """
    if spec is None:
        return dict(_DEFAULT_SPEC)
    if not isinstance(spec, dict):
        raise TypeError("observation_noise must be a mapping")

    enabled_explicit = "enabled" in spec
    normalized = dict(_DEFAULT_SPEC)
    normalized.update(deepcopy(spec))
    normalized["profile"] = str(normalized.get("profile") or NO_OBSERVATION_NOISE_PROFILE)
    normalized["interpretation"] = OBSERVATION_NOISE_INTERPRETATION

    for key in (
        "pose_noise_std_m",
        "heading_noise_std_rad",
        "lidar_dropout_prob",
        "lidar_dropout_value",
        "pedestrian_false_negative_prob",
        "pedestrian_false_positive_prob",
        "pedestrian_false_positive_radius_m",
        "pedestrian_false_positive_radius",
    ):
        normalized[key] = float(normalized.get(key, 0.0))

    for key in (
        "lidar_dropout_prob",
        "pedestrian_false_negative_prob",
        "pedestrian_false_positive_prob",
    ):
        value = float(normalized[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"observation_noise.{key} must be between 0 and 1")

    for key in (
        "pose_noise_std_m",
        "heading_noise_std_rad",
        "pedestrian_false_positive_radius_m",
        "pedestrian_false_positive_radius",
    ):
        if float(normalized[key]) < 0.0:
            raise ValueError(f"observation_noise.{key} must be non-negative")

    seed = normalized.get("seed")
    normalized["seed"] = int(seed) if seed is not None else None
    active = any(
        float(normalized[key]) > 0.0
        for key in (
            "pose_noise_std_m",
            "heading_noise_std_rad",
            "lidar_dropout_prob",
            "pedestrian_false_negative_prob",
            "pedestrian_false_positive_prob",
        )
    )
    normalized["enabled"] = (
        bool(normalized.get("enabled", False)) if enabled_explicit else active
    ) and active
    if not normalized["enabled"]:
        normalized["profile"] = NO_OBSERVATION_NOISE_PROFILE
    return normalized


def load_observation_noise_spec(path: str | Path) -> dict[str, Any]:
    """Load and normalize an observation-noise YAML file.

    Returns:
        Canonical observation-noise spec.
    """
    noise_path = Path(path)
    data = yaml.safe_load(noise_path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict) and isinstance(data.get("observation_noise"), dict):
        data = data["observation_noise"]
    return normalize_observation_noise_spec(data)


def make_observation_noise_rng(
    spec: dict[str, Any], *, seed: int, scenario_id: str
) -> np.random.Generator:
    """Create a deterministic per-episode RNG for observation corruption.

    Returns:
        NumPy generator scoped to the scenario seed and normalized spec.
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


def new_observation_noise_stats() -> dict[str, int]:
    """Return zeroed per-episode observation-noise counters."""
    return {
        "steps_with_noise": 0,
        "pose_noise_applied": 0,
        "heading_noise_applied": 0,
        "lidar_values_dropped": 0,
        "pedestrians_removed": 0,
        "pedestrians_added": 0,
    }


def apply_observation_noise(
    obs: dict[str, Any],
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Apply benchmark observation noise to one planner input observation.

    Returns:
        Pair of noisy observation and step-level noise counters.
    """
    stats = new_observation_noise_stats()
    if not bool(spec.get("enabled", False)):
        return obs, stats

    noisy = deepcopy(obs)
    _apply_pose_noise(noisy, spec, rng, stats)
    _apply_lidar_dropout(noisy, spec, rng, stats)
    _apply_pedestrian_noise(noisy, spec, rng, stats)
    if any(value > 0 for key, value in stats.items() if key != "steps_with_noise"):
        stats["steps_with_noise"] = 1
    return noisy, stats


def _as_xy_array(value: Any) -> np.ndarray | None:
    arr = np.asarray(value, dtype=float)
    if arr.size < 2:
        return None
    return arr.reshape(-1)[:2].astype(float, copy=True)


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
    elif "robot_heading" in obs:
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

    fp_prob = float(spec.get("pedestrian_false_positive_prob", 0.0))
    if fp_prob > 0.0 and float(rng.random()) < fp_prob:
        robot = obs.get("robot") if isinstance(obs.get("robot"), dict) else {}
        robot_pos = _as_xy_array(robot.get("position", [0.0, 0.0]))
        if robot_pos is None:
            robot_pos = np.zeros(2, dtype=float)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius_m = float(rng.uniform(0.0, spec.get("pedestrian_false_positive_radius_m", 4.0)))
        fp_pos = robot_pos + np.array([np.cos(angle), np.sin(angle)], dtype=float) * radius_m
        positions = np.vstack([positions, fp_pos.reshape(1, 2)])
        velocities = np.vstack([velocities, np.zeros((1, 2), dtype=float)])
        radii = np.concatenate(
            [radii, np.array([float(spec.get("pedestrian_false_positive_radius", 0.35))])]
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
    """Apply pedestrian noise to flattened SocNav structured observations."""
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

    fp_prob = float(spec.get("pedestrian_false_positive_prob", 0.0))
    if (
        fp_prob > 0.0
        and positions.shape[0] < positions_buf.shape[0]
        and float(rng.random()) < fp_prob
    ):
        robot_pos = _as_xy_array(obs.get("robot_position"))
        if robot_pos is None:
            robot = obs.get("robot") if isinstance(obs.get("robot"), dict) else {}
            robot_pos = _as_xy_array(robot.get("position", [0.0, 0.0]))
        if robot_pos is None:
            robot_pos = np.zeros(2, dtype=float)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius_m = float(rng.uniform(0.0, spec.get("pedestrian_false_positive_radius_m", 4.0)))
        fp_pos = robot_pos + np.array([np.cos(angle), np.sin(angle)], dtype=float) * radius_m
        positions = np.vstack([positions, fp_pos.reshape(1, 2)])
        velocities = np.vstack([velocities, np.zeros((1, 2), dtype=float)])
        stats["pedestrians_added"] += 1

    out_positions = np.array(positions_buf, copy=True)
    out_velocities = np.array(velocities_buf, copy=True)
    out_positions[:] = 0.0
    out_velocities[:] = 0.0
    out_count = min(positions.shape[0], out_positions.shape[0])
    if out_count:
        out_positions[:out_count] = positions[:out_count]
        out_velocities[:out_count] = velocities[:out_count]
    obs["pedestrians_positions"] = out_positions.tolist()
    obs["pedestrians_velocities"] = out_velocities.tolist()
    obs["pedestrians_count"] = np.array([float(out_count)], dtype=float).tolist()


def merge_observation_noise_stats(
    total: dict[str, int],
    step_stats: dict[str, int],
) -> dict[str, int]:
    """Accumulate step-level noise counters into an episode total.

    Returns:
        The updated total counter mapping.
    """
    for key, value in step_stats.items():
        total[key] = int(total.get(key, 0)) + int(value)
    return total
