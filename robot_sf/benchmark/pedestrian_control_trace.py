"""Per-pedestrian control-trace logging for heterogeneous-population analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from robot_sf.benchmark.finite_checks import require_finite_array, require_finite_scalar

PEDESTRIAN_CONTROL_TRACE_SCHEMA = "pedestrian-control-trace.v1"

_ARCHETYPE_KEYS = (
    "archetype",
    "pedestrian_archetype",
    "behavior_archetype",
    "speed_archetype",
)


def has_pedestrian_control_trace_metadata(scenario: Mapping[str, Any]) -> bool:
    """Return true when a scenario carries explicit per-pedestrian archetype labels."""

    single_pedestrians = _single_pedestrians(scenario)
    return any(_pedestrian_archetype(pedestrian) is not None for pedestrian in single_pedestrians)


def attach_pedestrian_control_trace(
    metadata: dict[str, Any],
    *,
    scenario: Mapping[str, Any],
    ped_positions: np.ndarray,
    ped_forces: np.ndarray | None,
    dt: float,
) -> None:
    """Attach a control trace to episode metadata when archetype labels are available.

    Args:
        metadata: Mutable algorithm metadata payload receiving `pedestrian_control_trace`.
        scenario: Scenario dictionary containing optional `single_pedestrians` metadata.
        ped_positions: Simulator pedestrian positions shaped `(steps, pedestrians, 2)`.
        ped_forces: Optional simulator pedestrian force vectors with the same shape.
        dt: Episode time step in seconds.
    """

    if not has_pedestrian_control_trace_metadata(scenario):
        return
    metadata["pedestrian_control_trace"] = build_pedestrian_control_trace(
        scenario=scenario,
        ped_positions=ped_positions,
        ped_forces=ped_forces,
        dt=dt,
    )


def build_pedestrian_control_trace(
    *,
    scenario: Mapping[str, Any],
    ped_positions: np.ndarray,
    ped_forces: np.ndarray | None,
    dt: float,
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
        pedestrian_steps: list[dict[str, Any]] = []
        previous_position: np.ndarray | None = None
        for step in range(step_count):
            position = positions[step, simulator_index]
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
            if forces is not None:
                force = forces[step, simulator_index]
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

        pedestrians.append(
            {
                "id": str(pedestrian_metadata.get("id") or f"pedestrian_{simulator_index}"),
                "simulator_index": simulator_index,
                "archetype": archetype,
                "steps": pedestrian_steps,
            }
        )

    return {
        "schema_version": PEDESTRIAN_CONTROL_TRACE_SCHEMA,
        "dt": dt_value,
        "source": "map_runner_episode",
        "archetype_source": "scenario.single_pedestrians.metadata",
        "pedestrian_count": pedestrian_count,
        "step_count": step_count,
        "pedestrians": pedestrians,
    }


def _single_pedestrians(scenario: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    single_pedestrians = scenario.get("single_pedestrians")
    if not isinstance(single_pedestrians, Sequence) or isinstance(single_pedestrians, str):
        return []
    return [ped for ped in single_pedestrians if isinstance(ped, Mapping)]


def _aligned_pedestrian_metadata(
    scenario: Mapping[str, Any],
    pedestrian_count: int,
) -> list[Mapping[str, Any]]:
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


__all__ = [
    "PEDESTRIAN_CONTROL_TRACE_SCHEMA",
    "attach_pedestrian_control_trace",
    "build_pedestrian_control_trace",
    "has_pedestrian_control_trace_metadata",
]
