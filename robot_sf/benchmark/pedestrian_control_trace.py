"""Per-pedestrian control-trace logging for heterogeneous-population analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from robot_sf.benchmark.finite_checks import require_finite_array, require_finite_scalar

PEDESTRIAN_CONTROL_TRACE_SCHEMA = "pedestrian-control-trace.v1"
PEDESTRIAN_CONTROL_TRACE_LABELS_SCHEMA = "pedestrian-control-trace-labels.v1"
PEDESTRIAN_CONTROL_TRACE_LABELS_KEY = "pedestrian_control_trace_labels"

_ARCHETYPE_KEYS = (
    "archetype",
    "pedestrian_archetype",
    "behavior_archetype",
    "speed_archetype",
)
_OPTIONAL_LABEL_KEYS = ("desired_speed_factor", "response_law", "source")


def has_pedestrian_control_trace_metadata(scenario: Mapping[str, Any]) -> bool:
    """Return true when a scenario carries explicit per-pedestrian archetype labels."""

    if PEDESTRIAN_CONTROL_TRACE_LABELS_KEY in scenario:
        trace_labels = _trace_label_records(scenario)
        return any(_pedestrian_archetype(pedestrian) is not None for pedestrian in trace_labels)
    single_pedestrians = _single_pedestrians(scenario)
    return any(_pedestrian_archetype(pedestrian) is not None for pedestrian in single_pedestrians)


def attach_pedestrian_control_trace(
    metadata: dict[str, Any],
    *,
    scenario: Mapping[str, Any],
    ped_positions: np.ndarray,
    ped_forces: np.ndarray | None,
    dt: float,
    robot_positions: np.ndarray | None = None,
    robot_radius: float = 0.0,
    ped_radius: float = 0.0,
) -> None:
    """Attach a control trace to episode metadata when archetype labels are available.

    Args:
        metadata: Mutable algorithm metadata payload receiving `pedestrian_control_trace`.
        scenario: Scenario dictionary containing optional `single_pedestrians` metadata.
            Generated-population scenarios may instead provide
            `pedestrian_control_trace_labels`.
        ped_positions: Simulator pedestrian positions shaped `(steps, pedestrians, 2)`.
        ped_forces: Optional simulator pedestrian force vectors with the same shape.
        dt: Episode time step in seconds.
        robot_positions: Optional simulator robot positions shaped `(steps, 2)`.
        robot_radius: Optional simulator robot footprint radius.
        ped_radius: Optional simulator pedestrian footprint radius.
    """

    if not has_pedestrian_control_trace_metadata(scenario):
        return
    metadata["pedestrian_control_trace"] = build_pedestrian_control_trace(
        scenario=scenario,
        ped_positions=ped_positions,
        ped_forces=ped_forces,
        dt=dt,
        robot_positions=robot_positions,
        robot_radius=robot_radius,
        ped_radius=ped_radius,
    )


def _validate_robot_parameters(
    robot_positions: np.ndarray | None,
    robot_radius: float,
    ped_radius: float,
    positions: np.ndarray,
) -> tuple[np.ndarray | None, float, float]:
    """Validate robot parameters.

    Returns:
        tuple[np.ndarray | None, float, float]: Resolved position, robot_radius, and ped_radius.
    """
    if robot_positions is None:
        return None, 0.0, 0.0
    robot_pos = require_finite_array("pedestrian_control_trace.robot_positions", robot_positions)
    if robot_pos.ndim != 2 or robot_pos.shape[1] != 2:
        raise ValueError(
            "pedestrian_control_trace.robot_positions must have shape (steps, 2)"
        )
    if robot_pos.shape[0] != positions.shape[0]:
        raise ValueError(
            "pedestrian_control_trace.robot_positions step count must match ped_positions"
        )
    robot_rad = require_finite_scalar("pedestrian_control_trace.robot_radius", robot_radius)
    if robot_rad < 0.0:
        raise ValueError("robot_radius must be non-negative")
    ped_rad = require_finite_scalar("pedestrian_control_trace.ped_radius", ped_radius)
    if ped_rad < 0.0:
        raise ValueError("ped_radius must be non-negative")
    return robot_pos, robot_rad, ped_rad


def _build_pedestrian_steps(
    *,
    positions: np.ndarray,
    forces: np.ndarray | None,
    robot_pos: np.ndarray | None,
    robot_rad: float,
    ped_rad: float,
    dt_value: float,
    simulator_index: int,
) -> list[dict[str, Any]]:
    """Build per-step payloads for a single pedestrian.

    Returns:
        list[dict[str, Any]]: Per-step payloads for the pedestrian.
    """
    step_count = int(positions.shape[0])
    pedestrian_steps: list[dict[str, Any]] = []
    previous_position: np.ndarray | None = None
    for step in range(step_count):
        position = positions[step]
        velocity = (
            (position - previous_position) / dt_value
            if previous_position is not None
            else np.zeros(2, dtype=float)
        )
        speed = float(np.linalg.norm(velocity))
        require_finite_scalar(
            f"pedestrian_control_trace.pedestrians[{simulator_index}].steps[{step}].speed_m_s",
            speed,
        )
        payload: dict[str, Any] = {
            "step": step,
            "x_m": float(position[0]),
            "y_m": float(position[1]),
            "vx_m_s": float(velocity[0]),
            "vy_m_s": float(velocity[1]),
            "speed_m_s": speed,
        }
        if robot_pos is not None:
            r_pos = robot_pos[step]
            dist = float(np.linalg.norm(position - r_pos))
            clearance = dist - (robot_rad + ped_rad)
            require_finite_scalar(
                f"pedestrian_control_trace.pedestrians[{simulator_index}].steps[{step}].clearance_m",
                clearance,
            )
            payload["clearance_m"] = clearance
        if forces is not None:
            force = forces[step]
            force_norm = float(np.linalg.norm(force))
            require_finite_scalar(
                "pedestrian_control_trace."
                f"pedestrians[{simulator_index}].steps[{step}].force_norm",
                force_norm,
            )
            payload.update(
                {
                    "force_x": float(force[0]),
                    "force_y": float(force[1]),
                    "force_norm": force_norm,
                }
            )
        pedestrian_steps.append(payload)
        previous_position = np.array(position, dtype=float, copy=True)
    return pedestrian_steps


def build_pedestrian_control_trace(
    *,
    scenario: Mapping[str, Any],
    ped_positions: np.ndarray,
    ped_forces: np.ndarray | None,
    dt: float,
    robot_positions: np.ndarray | None = None,
    robot_radius: float = 0.0,
    ped_radius: float = 0.0,
) -> dict[str, Any]:
    """Build a finite, per-pedestrian control trace grouped by simulator pedestrian index.

    Returns:
        Versioned JSON-serializable pedestrian control trace payload.

    Raises:
        ValueError: Missing archetype metadata, shape mismatch, or non-finite trace values.
    """

    dt_value = require_finite_scalar("pedestrian_control_trace.dt", dt)
    if dt_value <= 0.0:
        raise ValueError("pedestrian_control_trace.dt must be positive")

    positions = require_finite_array("pedestrian_control_trace.ped_positions", ped_positions)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(
            "pedestrian_control_trace.ped_positions must have shape (steps, pedestrians, 2)"
        )

    forces: np.ndarray | None = None
    if ped_forces is not None:
        forces = require_finite_array("pedestrian_control_trace.ped_forces", ped_forces)
        if forces.shape != positions.shape:
            raise ValueError("pedestrian_control_trace.ped_forces must match ped_positions shape")

    robot_pos, robot_rad, ped_rad = _validate_robot_parameters(
        robot_positions, robot_radius, ped_radius, positions
    )

    step_count = int(positions.shape[0])
    pedestrian_count = int(positions.shape[1])
    metadata = _aligned_pedestrian_metadata(scenario, pedestrian_count)
    pedestrians: list[dict[str, Any]] = []
    for simulator_index, pedestrian_metadata in enumerate(metadata):
        archetype = _pedestrian_archetype(pedestrian_metadata)
        if archetype is None:
            raise ValueError(
                "pedestrian_control_trace requires archetype metadata for "
                f"simulator pedestrian {simulator_index}"
            )
        pedestrian_steps = _build_pedestrian_steps(
            positions=positions[:, simulator_index],
            forces=forces[:, simulator_index] if forces is not None else None,
            robot_pos=robot_pos,
            robot_rad=robot_rad,
            ped_rad=ped_rad,
            dt_value=dt_value,
            simulator_index=simulator_index,
        )

        pedestrian_payload: dict[str, Any] = {
            "id": str(pedestrian_metadata.get("id") or f"pedestrian_{simulator_index}"),
            "simulator_index": simulator_index,
            "archetype": archetype,
            "steps": pedestrian_steps,
        }
        _copy_optional_label_fields(pedestrian_payload, pedestrian_metadata)
        pedestrians.append(pedestrian_payload)

    archetype_source = (
        f"scenario.{PEDESTRIAN_CONTROL_TRACE_LABELS_KEY}"
        if PEDESTRIAN_CONTROL_TRACE_LABELS_KEY in scenario
        else "scenario.single_pedestrians.metadata"
    )
    return {
        "schema_version": PEDESTRIAN_CONTROL_TRACE_SCHEMA,
        "dt": dt_value,
        "source": "map_runner_episode",
        "archetype_source": archetype_source,
        "pedestrian_count": pedestrian_count,
        "step_count": step_count,
        "pedestrians": pedestrians,
    }


def build_generated_population_control_trace_labels(
    population_records: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    """Build stable simulator-index labels for generated heterogeneous populations.

    Attach returned records to a benchmark scenario as
    ``pedestrian_control_trace_labels`` before map-runner execution. The records
    carry no metric values; they only label generated pedestrians so the
    mean-matched harness can distinguish ready traces from exact missing fields.

    Returns:
        Stable label records sorted by simulator pedestrian index.
    """

    source_value = str(source).strip()
    if not source_value:
        raise ValueError("pedestrian_control_trace_labels source must be non-empty")
    labels: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for record_index, record in enumerate(population_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"population_records[{record_index}] must be mapping")
        try:
            simulator_index = int(record["simulator_index"])
        except KeyError as exc:
            raise ValueError(f"population_records[{record_index}].simulator_index missing") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"population_records[{record_index}].simulator_index must be integer"
            ) from exc
        if simulator_index < 0:
            raise ValueError(f"population_records[{record_index}].simulator_index must be >= 0")
        if simulator_index in seen_indices:
            raise ValueError(
                f"pedestrian_control_trace_labels duplicate simulator_index {simulator_index}"
            )
        seen_indices.add(simulator_index)
        archetype = _pedestrian_archetype(record)
        if archetype is None:
            raise ValueError(f"population_records[{record_index}] missing archetype")
        label: dict[str, Any] = {
            "schema_version": PEDESTRIAN_CONTROL_TRACE_LABELS_SCHEMA,
            "id": str(record.get("id") or f"pedestrian_{simulator_index}"),
            "simulator_index": simulator_index,
            "archetype": archetype,
            "source": source_value,
        }
        _copy_optional_label_fields(label, record)
        labels.append(label)
    labels.sort(key=lambda item: int(item["simulator_index"]))
    return labels


def _single_pedestrians(scenario: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    single_pedestrians = scenario.get("single_pedestrians")
    if not isinstance(single_pedestrians, Sequence) or isinstance(single_pedestrians, str):
        return []
    return [ped for ped in single_pedestrians if isinstance(ped, Mapping)]


def _trace_label_records(scenario: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if PEDESTRIAN_CONTROL_TRACE_LABELS_KEY not in scenario:
        return []
    labels = scenario.get(PEDESTRIAN_CONTROL_TRACE_LABELS_KEY)
    if not isinstance(labels, Sequence) or isinstance(labels, str):
        raise ValueError(f"scenario.{PEDESTRIAN_CONTROL_TRACE_LABELS_KEY} must be sequence")
    trace_labels: list[Mapping[str, Any]] = []
    for label_index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise ValueError(
                f"scenario.{PEDESTRIAN_CONTROL_TRACE_LABELS_KEY}[{label_index}] must be mapping"
            )
        trace_labels.append(label)
    return trace_labels


def _aligned_pedestrian_metadata(
    scenario: Mapping[str, Any],
    pedestrian_count: int,
) -> list[Mapping[str, Any]]:
    if PEDESTRIAN_CONTROL_TRACE_LABELS_KEY in scenario:
        trace_labels = _trace_label_records(scenario)
        return _aligned_trace_label_metadata(trace_labels, pedestrian_count)

    single_pedestrians = _single_pedestrians(scenario)
    if pedestrian_count == 0:
        return []
    if not single_pedestrians:
        raise ValueError("pedestrian_control_trace requires scenario.single_pedestrians metadata")
    if len(single_pedestrians) > pedestrian_count:
        raise ValueError(
            "pedestrian_control_trace single_pedestrians metadata exceeds simulator pedestrian count"
        )
    offset = pedestrian_count - len(single_pedestrians)
    metadata: list[Mapping[str, Any]] = [{} for _ in range(offset)]
    metadata.extend(single_pedestrians)
    return metadata


def _aligned_trace_label_metadata(
    trace_labels: Sequence[Mapping[str, Any]],
    pedestrian_count: int,
) -> list[Mapping[str, Any]]:
    if pedestrian_count == 0:
        return []
    if len(trace_labels) != pedestrian_count:
        raise ValueError(
            "pedestrian_control_trace_labels length must equal simulator pedestrian count "
            f"(got {len(trace_labels)}, expected {pedestrian_count})"
        )
    by_index: dict[int, Mapping[str, Any]] = {}
    for label_index, label in enumerate(trace_labels):
        try:
            simulator_index = int(label["simulator_index"])
        except KeyError as exc:
            raise ValueError(
                f"pedestrian_control_trace_labels[{label_index}].simulator_index missing"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"pedestrian_control_trace_labels[{label_index}].simulator_index must be integer"
            ) from exc
        if not 0 <= simulator_index < pedestrian_count:
            raise ValueError(
                f"pedestrian_control_trace_labels[{label_index}].simulator_index out of range"
            )
        if simulator_index in by_index:
            raise ValueError(
                f"pedestrian_control_trace_labels duplicate simulator_index {simulator_index}"
            )
        by_index[simulator_index] = label
    return [by_index[index] for index in range(pedestrian_count)]


def _pedestrian_archetype(pedestrian: Mapping[str, Any]) -> str | None:
    metadata = pedestrian.get("metadata") if isinstance(pedestrian.get("metadata"), Mapping) else {}
    for source in (metadata, pedestrian):
        for key in _ARCHETYPE_KEYS:
            value = source.get(key)
            if value is None:
                continue
            label = str(value).strip()
            if label:
                return label
    speed = pedestrian.get("speed_m_s")
    if isinstance(speed, int | float | np.integer | np.floating):
        speed_value = float(speed)
        if math.isfinite(speed_value):
            return f"speed_m_s:{speed_value:g}"
    return None


def _copy_optional_label_fields(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key in _OPTIONAL_LABEL_KEYS:
        if key in source and source[key] is not None:
            destination[key] = source[key]


__all__ = [
    "PEDESTRIAN_CONTROL_TRACE_LABELS_KEY",
    "PEDESTRIAN_CONTROL_TRACE_LABELS_SCHEMA",
    "PEDESTRIAN_CONTROL_TRACE_SCHEMA",
    "attach_pedestrian_control_trace",
    "build_generated_population_control_trace_labels",
    "build_pedestrian_control_trace",
    "has_pedestrian_control_trace_metadata",
]
