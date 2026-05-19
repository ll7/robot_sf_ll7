"""Graded benchmark observation-level contracts.

Observation levels describe perception assumptions separately from raw planner
observation modes. They are benchmark metadata and compatibility gates, not
camera or detector implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObservationLevelSpec:
    """Benchmark-facing perception assumption for planner observations."""

    key: str
    display_name: str
    perception_assumption: str
    compatible_observation_modes: tuple[str, ...]
    required_inputs: tuple[str, ...]
    noise_policy: str = "none"
    occlusion_policy: str = "none"
    interpretation: str = "benchmark_metadata_not_sensor_certification"

    def to_metadata(self, *, active_observation_mode: str | None = None) -> dict[str, Any]:
        """Return a JSON-serializable observation-level payload.

        Returns:
            Metadata describing the observation-level contract.
        """
        return {
            "key": self.key,
            "display_name": self.display_name,
            "perception_assumption": self.perception_assumption,
            "active_observation_mode": active_observation_mode,
            "compatible_observation_modes": list(self.compatible_observation_modes),
            "required_inputs": list(self.required_inputs),
            "noise_policy": self.noise_policy,
            "occlusion_policy": self.occlusion_policy,
            "interpretation": self.interpretation,
        }


_OBSERVATION_LEVELS: dict[str, ObservationLevelSpec] = {
    "oracle_full_state": ObservationLevelSpec(
        key="oracle_full_state",
        display_name="Oracle full state",
        perception_assumption="privileged_sim_state",
        compatible_observation_modes=(
            "goal_state",
            "socnav_state",
            "headed_socnav_state",
            "gst_human_state",
        ),
        required_inputs=("robot_state", "goal", "pedestrians"),
    ),
    "tracked_agents_no_noise": ObservationLevelSpec(
        key="tracked_agents_no_noise",
        display_name="Perfect tracked agents",
        perception_assumption="perfect_tracking_without_noise",
        compatible_observation_modes=(
            "socnav_state",
            "headed_socnav_state",
            "gst_human_state",
        ),
        required_inputs=("robot_state", "goal", "tracked_agents"),
    ),
    "tracked_agents_with_noise": ObservationLevelSpec(
        key="tracked_agents_with_noise",
        display_name="Noisy tracked agents",
        perception_assumption="tracked_agents_with_synthetic_noise",
        compatible_observation_modes=(
            "socnav_state",
            "headed_socnav_state",
            "gst_human_state",
        ),
        required_inputs=("robot_state", "goal", "tracked_agents"),
        noise_policy="benchmark_observation_noise",
        interpretation="synthetic_noise_robustness_not_real_sensor_calibration",
    ),
    "lidar_2d": ObservationLevelSpec(
        key="lidar_2d",
        display_name="2D lidar projection",
        perception_assumption="range_sensor_projection",
        compatible_observation_modes=("sensor_fusion_state", "lidar_human_state"),
        required_inputs=("robot_state", "goal", "lidar_rays"),
    ),
    "occluded_partial_state": ObservationLevelSpec(
        key="occluded_partial_state",
        display_name="Occluded partial state",
        perception_assumption="partial_state_with_occlusion",
        compatible_observation_modes=("socnav_state", "sensor_fusion_state"),
        required_inputs=("robot_state", "goal", "visible_agents"),
        occlusion_policy="scenario_visibility_or_sensor_mask",
        interpretation="occlusion_metadata_not_sim_to_real_validity",
    ),
}

OBSERVATION_LEVEL_KEYS = tuple(_OBSERVATION_LEVELS)
DEFAULT_OBSERVATION_LEVEL = "oracle_full_state"


def observation_level_spec(level: str | None = None) -> ObservationLevelSpec:
    """Return a normalized observation-level specification.

    Returns:
        Observation-level specification for ``level``.
    """
    key = str(level or DEFAULT_OBSERVATION_LEVEL).strip().lower()
    try:
        return _OBSERVATION_LEVELS[key]
    except KeyError as exc:
        supported = ", ".join(OBSERVATION_LEVEL_KEYS)
        raise ValueError(
            f"Unsupported observation level '{key}'. Supported levels: {supported}."
        ) from exc


def observation_level_for_mode(observation_mode: str) -> ObservationLevelSpec:
    """Infer a default observation-level label for a legacy observation mode.

    Returns:
        Best-effort observation-level specification for ``observation_mode``.
    """
    mode = str(observation_mode).strip()
    if mode in {"sensor_fusion_state", "lidar_human_state"}:
        return observation_level_spec("lidar_2d")
    if mode in {"socnav_state", "headed_socnav_state", "gst_human_state"}:
        return observation_level_spec("tracked_agents_no_noise")
    return observation_level_spec(DEFAULT_OBSERVATION_LEVEL)


def resolve_observation_level_contract(
    algo: str,
    *,
    observation_level: str | None = None,
    requested_observation_mode: str | None = None,
    algorithm_default_mode: str | None = None,
    algorithm_supported_modes: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resolve the active observation mode implied by an observation level.

    Returns:
        Payload with ``observation_level`` metadata and ``active_observation_mode``.
    """
    level = observation_level_spec(observation_level)
    supported = tuple(str(mode) for mode in algorithm_supported_modes)
    level_modes = tuple(level.compatible_observation_modes)
    candidate_modes = supported or level_modes

    requested = str(requested_observation_mode).strip() if requested_observation_mode else ""
    if requested:
        if supported and requested not in supported:
            supported_text = ", ".join(supported)
            raise ValueError(
                f"Observation mode '{requested}' is not supported by algorithm '{algo}'. "
                f"Supported modes: {supported_text}."
            )
        if requested not in level_modes:
            level_text = ", ".join(level_modes)
            raise ValueError(
                f"Observation mode '{requested}' is not compatible with observation level "
                f"'{level.key}'. Compatible modes: {level_text}."
            )
        active_mode = requested
    else:
        default_mode = str(algorithm_default_mode or "").strip()
        if default_mode and default_mode in candidate_modes and default_mode in level_modes:
            active_mode = default_mode
        else:
            overlap = [mode for mode in candidate_modes if mode in level_modes]
            if not overlap:
                level_text = ", ".join(level_modes)
                supported_text = ", ".join(supported)
                raise ValueError(
                    f"Observation level '{level.key}' is not supported by algorithm '{algo}'. "
                    f"Level modes: {level_text}. Algorithm modes: {supported_text}."
                )
            active_mode = overlap[0]

    return {
        "observation_level": level.to_metadata(active_observation_mode=active_mode),
        "active_observation_mode": active_mode,
    }


__all__ = [
    "DEFAULT_OBSERVATION_LEVEL",
    "OBSERVATION_LEVEL_KEYS",
    "ObservationLevelSpec",
    "observation_level_for_mode",
    "observation_level_spec",
    "resolve_observation_level_contract",
]
