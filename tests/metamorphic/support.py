"""Small deterministic fixtures and comparison helpers for metamorphic tests.

The fixture deliberately uses :class:`CrowdSimEnv` with explicit pedestrians and no
obstacle force.  This keeps each relation focused on a single environment contract
without making a benchmark or changing simulator behavior.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.gym_env.crowd_sim_env import CrowdSimEnv, CrowdSimulationConfig
from robot_sf.gym_env.env_config import SimulationSettings
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition

if TYPE_CHECKING:
    from pathlib import Path

SCENE_SIZE = 20.0
EPISODE_SEED = 8244
EPISODE_STEPS = 3
TRACE_ATOL = 1e-5
OBSERVATION_FIELDS = ("positions", "velocities", "goals", "forces", "step")
VECTOR_FIELDS = frozenset(("positions", "velocities", "goals", "forces"))

_BASE_PEDESTRIANS = (
    ("north", (4.0, 5.0), (16.0, 5.0)),
    ("middle", (16.0, 10.0), (4.0, 10.0)),
    ("south", (4.0, 15.0), (16.0, 15.0)),
)
_BASE_BOUNDS = (
    ((0.0, 0.0), (SCENE_SIZE, 0.0)),
    ((SCENE_SIZE, 0.0), (SCENE_SIZE, SCENE_SIZE)),
    ((SCENE_SIZE, SCENE_SIZE), (0.0, SCENE_SIZE)),
    ((0.0, SCENE_SIZE), (0.0, 0.0)),
)
_DUMMY_ZONE = ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0))

PointTransform = Callable[[tuple[float, float]], tuple[float, float]]
ObservationTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EpisodeTrace:
    """Copied observations and row identities from one bounded episode."""

    observations: tuple[dict[str, np.ndarray], ...]
    row_keys: tuple[str, ...]
    infos: tuple[dict[str, Any], ...]


def _point(x: float, y: float) -> tuple[float, float]:
    """Return a plain finite point for map-definition construction."""
    return (float(x), float(y))


def _pedestrians(
    transform: PointTransform | None = None,
    *,
    order: Sequence[int] | None = None,
) -> list[SinglePedestrianDefinition]:
    """Build explicit pedestrians, optionally transforming and reordering rows."""
    point_transform = transform or (lambda point: point)
    indices = tuple(range(len(_BASE_PEDESTRIANS))) if order is None else tuple(order)
    result = []
    for index in indices:
        pedestrian_id, start, goal = _BASE_PEDESTRIANS[index]
        result.append(
            SinglePedestrianDefinition(
                id=pedestrian_id,
                start=point_transform(start),
                goal=point_transform(goal),
                speed_m_s=1.0,
            )
        )
    return result


def _transform_zone(
    zone: Sequence[tuple[float, float]],
    transform: PointTransform,
) -> tuple[tuple[float, float], ...]:
    """Transform a polygon-like zone into immutable point tuples."""
    return tuple(transform(point) for point in zone)


def _build_map(
    transform: PointTransform | None = None,
    *,
    order: Sequence[int] | None = None,
) -> MapDefinition:
    """Build the synthetic map used by all environment-level relations."""
    point_transform = transform or (lambda point: point)
    return MapDefinition(
        width=SCENE_SIZE,
        height=SCENE_SIZE,
        obstacles=[],
        robot_spawn_zones=[_transform_zone(_DUMMY_ZONE, point_transform)],
        ped_spawn_zones=[],
        robot_goal_zones=[_transform_zone(_DUMMY_ZONE, point_transform)],
        bounds=[_transform_zone(bound, point_transform) for bound in _BASE_BOUNDS],
        robot_routes=[],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=_pedestrians(point_transform, order=order),
    )


BASE_MAP = _build_map()


def translated_map(offset: Sequence[float]) -> MapDefinition:
    """Return a copy of the fixture translated by ``offset`` in global XY."""
    delta = np.asarray(offset, dtype=float)
    if delta.shape != (2,):
        raise ValueError(f"offset must have shape (2,), got {delta.shape}")

    def translate(point: tuple[float, float]) -> tuple[float, float]:
        return _point(point[0] + delta[0], point[1] + delta[1])

    return _build_map(translate)


def rotated_map() -> MapDefinition:
    """Return the fixture rotated 90 degrees counter-clockwise around its center."""
    center = np.asarray((SCENE_SIZE / 2.0, SCENE_SIZE / 2.0), dtype=float)

    def rotate(point: tuple[float, float]) -> tuple[float, float]:
        centered = np.asarray(point, dtype=float) - center
        return _point(center[0] - centered[1], center[1] + centered[0])

    return _build_map(rotate)


def permuted_map() -> MapDefinition:
    """Return the fixture with its declared pedestrian rows reversed."""
    return _build_map(order=tuple(reversed(range(len(_BASE_PEDESTRIANS)))))


def _settings(*, oracle_enabled: bool) -> SimulationSettings:
    """Return short, fixed-step settings with no generated background population."""
    population = len(_BASE_PEDESTRIANS)
    return SimulationSettings(
        sim_time_in_secs=(EPISODE_STEPS + 1) * 0.1,
        time_per_step_in_secs=0.1,
        max_total_pedestrians=population,
        population_size=population,
        oracle_force_trace_enabled=oracle_enabled,
    )


def make_env(
    map_def: MapDefinition = BASE_MAP,
    *,
    render_mode: str | None = None,
    recording_path: Path | None = None,
    oracle_enabled: bool = False,
) -> CrowdSimEnv:
    """Create a deterministic crowd environment around one synthetic map."""
    map_pool = MapDefinitionPool(map_defs={"synthetic": map_def})
    config = CrowdSimulationConfig(
        sim_config=_settings(oracle_enabled=oracle_enabled),
        map_pool=map_pool,
        map_id="synthetic",
        peds_have_obstacle_forces=False,
        render_mode=render_mode,
        recording_enabled=recording_path is not None,
        recording_path=str(recording_path) if recording_path is not None else None,
    )
    return CrowdSimEnv(config)


def _copy_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Copy each environment array so later simulator steps cannot mutate a trace."""
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def capture_episode(
    env: CrowdSimEnv,
    *,
    row_keys: Sequence[str],
    steps: int = EPISODE_STEPS,
    seed: int = EPISODE_SEED,
) -> EpisodeTrace:
    """Capture reset plus a fixed number of automatic crowd steps."""
    observation, info = env.reset(seed=seed, options={"map_id": "synthetic"})
    observations = [_copy_observation(observation)]
    infos = [dict(info)]
    for _ in range(steps):
        observation, _reward, _terminated, _truncated, info = env.step()
        observations.append(_copy_observation(observation))
        infos.append(dict(info))
    return EpisodeTrace(tuple(observations), tuple(row_keys), tuple(infos))


def run_episode(
    map_def: MapDefinition = BASE_MAP,
    *,
    render_mode: str | None = None,
    recording_path: Path | None = None,
    oracle_enabled: bool = False,
    steps: int = EPISODE_STEPS,
    seed: int = EPISODE_SEED,
) -> EpisodeTrace:
    """Run and close a deterministic episode, returning only copied trace data."""
    env = make_env(
        map_def,
        render_mode=render_mode,
        recording_path=recording_path,
        oracle_enabled=oracle_enabled,
    )
    try:
        return capture_episode(
            env,
            row_keys=tuple(pedestrian.id for pedestrian in map_def.single_pedestrians),
            steps=steps,
            seed=seed,
        )
    finally:
        env.close()


def _max_abs_error(expected: np.ndarray, actual: np.ndarray) -> float:
    """Return a finite mismatch summary, using infinity for incompatible shapes."""
    if expected.shape != actual.shape:
        return float("inf")
    if expected.size == 0:
        return 0.0
    return float(np.max(np.abs(expected.astype(float) - actual.astype(float))))


def assert_trace_equal(
    expected: EpisodeTrace,
    actual: EpisodeTrace,
    *,
    row_order: Sequence[int] | None = None,
    transforms: dict[str, ObservationTransform] | None = None,
    atol: float = TRACE_ATOL,
) -> None:
    """Compare traces and report the first divergent step, field, and max error."""
    if len(expected.observations) != len(actual.observations):
        raise AssertionError(
            "first divergence: "
            f"step={min(len(expected.observations), len(actual.observations))} "
            "field=trace_length max_abs_error=inf"
        )
    transforms = transforms or {}
    matrix_order = None if row_order is None else np.asarray(tuple(row_order), dtype=int)
    for step_index, (expected_observation, actual_observation) in enumerate(
        zip(expected.observations, actual.observations, strict=True)
    ):
        for field in OBSERVATION_FIELDS:
            if field not in expected_observation or field not in actual_observation:
                raise AssertionError(
                    f"first divergence: step={step_index} field={field} max_abs_error=inf"
                )
            expected_values = np.asarray(expected_observation[field])
            actual_values = np.asarray(actual_observation[field])
            if matrix_order is not None and field in VECTOR_FIELDS:
                actual_values = actual_values[matrix_order]
            transform = transforms.get(field)
            if transform is not None:
                actual_values = np.asarray(transform(actual_values))
            shapes_match = expected_values.shape == actual_values.shape
            values_match = shapes_match and np.allclose(
                expected_values,
                actual_values,
                rtol=0.0,
                atol=atol,
            )
            if not values_match:
                error = _max_abs_error(expected_values, actual_values)
                raise AssertionError(
                    f"first divergence: step={step_index} field={field} max_abs_error={error:.9g}"
                )


def read_recording(path: Path, *, row_keys: Sequence[str]) -> tuple[list[str], EpisodeTrace]:
    """Read a compact ``CrowdSimEnv`` JSONL recording as a replayable state trace."""
    events: list[str] = []
    observations: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        events.append(str(payload["event"]))
        observations.append(
            {
                key: np.asarray(
                    value,
                    dtype=np.int64 if key == "step" else np.float32,
                )
                for key, value in payload["observation"].items()
            }
        )
        infos.append(dict(payload.get("info", {})))
    return events, EpisodeTrace(tuple(observations), tuple(row_keys), tuple(infos))
