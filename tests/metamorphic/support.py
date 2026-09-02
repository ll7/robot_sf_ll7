"""Small deterministic fixtures and comparison helpers for metamorphic tests.

The fixture deliberately uses :class:`CrowdSimEnv` with explicit pedestrians and no
obstacle force.  This keeps each relation focused on a single environment contract
without making a benchmark or changing simulator behavior.
"""

from __future__ import annotations

import hashlib
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
    identity_suffix: str = "",
) -> list[SinglePedestrianDefinition]:
    """Build explicit pedestrians, optionally transforming, reordering, or relabeling rows."""
    point_transform = transform or (lambda point: point)
    indices = tuple(range(len(_BASE_PEDESTRIANS))) if order is None else tuple(order)
    result = []
    for index in indices:
        pedestrian_id, start, goal = _BASE_PEDESTRIANS[index]
        result.append(
            SinglePedestrianDefinition(
                id=f"{pedestrian_id}{identity_suffix}",
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
    identity_suffix: str = "",
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
        single_pedestrians=_pedestrians(
            point_transform,
            order=order,
            identity_suffix=identity_suffix,
        ),
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


def relabeled_map() -> MapDefinition:
    """Return the fixture with simulator-only pedestrian labels randomized."""
    return _build_map(identity_suffix="-randomized")


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


def _canonical_json(value: Any) -> Any:
    """Return a strictly JSON-encodable structure with sorted, typed containers."""
    if isinstance(value, dict):
        return {str(key): _canonical_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256_c_order": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        }
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"info value {type(value).__name__} is not canonically encodable")


def _array_identity(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    """Return dtype string, shape, and C-order raw bytes of one array."""
    contiguous = np.ascontiguousarray(np.asarray(array))
    return contiguous.dtype.str, tuple(contiguous.shape), contiguous.tobytes(order="C")


def _first_byte_mismatch(expected_bytes: bytes, actual_bytes: bytes) -> int | None:
    """Return the first differing byte offset, or None when identical."""
    for offset, (expected_byte, actual_byte) in enumerate(
        zip(expected_bytes, actual_bytes, strict=False)
    ):
        if expected_byte != actual_byte:
            return offset
    if len(expected_bytes) != len(actual_bytes):
        return min(len(expected_bytes), len(actual_bytes))
    return None


def assert_trace_byte_identical(
    expected: EpisodeTrace,
    actual: EpisodeTrace,
    *,
    compare_infos: bool = True,
) -> None:
    """Require exact representation identity of actor-visible outputs.

    Unlike :func:`assert_trace_equal`, this comparison is byte-exact: the
    observation dtype, shape, and C-order byte sequence must match, and the
    JSON-serializable info payload must serialize identically. Numeric
    closeness never substitutes for representation identity here.
    """
    if len(expected.observations) != len(actual.observations):
        raise AssertionError("byte identity divergence: trace_length differs")
    for step_index, (expected_obs, actual_obs) in enumerate(
        zip(expected.observations, actual.observations, strict=True)
    ):
        expected_fields = set(expected_obs)
        actual_fields = set(actual_obs)
        if expected_fields != actual_fields:
            raise AssertionError(
                "byte identity divergence: "
                f"step={step_index} "
                f"fields_expected={sorted(expected_fields)} "
                f"fields_actual={sorted(actual_fields)}"
            )
        for field in sorted(expected_fields):
            expected_dtype, expected_shape, expected_bytes = _array_identity(expected_obs[field])
            actual_dtype, actual_shape, actual_bytes = _array_identity(actual_obs[field])
            mismatch = (
                expected_dtype != actual_dtype
                or expected_shape != actual_shape
                or _first_byte_mismatch(expected_bytes, actual_bytes) is not None
            )
            if mismatch:
                offset = (
                    _first_byte_mismatch(expected_bytes, actual_bytes)
                    if expected_dtype == actual_dtype and expected_shape == actual_shape
                    else None
                )
                raise AssertionError(
                    "byte identity divergence: "
                    f"step={step_index} field={field} "
                    f"dtype_expected={expected_dtype} dtype_actual={actual_dtype} "
                    f"shape_expected={expected_shape} shape_actual={actual_shape} "
                    f"first_differing_byte={offset}"
                )
        if not compare_infos:
            continue
        expected_info = expected.infos[step_index]
        actual_info = actual.infos[step_index]
        try:
            expected_payload = json.dumps(
                _canonical_json(dict(expected_info)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            actual_payload = json.dumps(
                _canonical_json(dict(actual_info)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise AssertionError(
                f"byte identity divergence: step={step_index} info not canonically encodable: {exc}"
            ) from exc
        if expected_payload != actual_payload:
            raise AssertionError(
                "byte identity divergence: "
                f"step={step_index} field=info "
                f"expected={expected_payload[:200]} actual={actual_payload[:200]}"
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
