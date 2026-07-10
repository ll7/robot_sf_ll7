"""Planner command-space and kinematics-feasibility helpers for benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.algorithm_metadata import planner_contract_for_algorithm

if TYPE_CHECKING:
    from robot_sf.planner.kinematics_model import KinematicsModel
from robot_sf.errors import RobotSfError

_DEFAULT_KINEMATICS = "differential_drive"


class PlannerContractValidationError(RobotSfError, ValueError):
    """Raised when planner observation/action metadata does not match a run request."""


def default_robot_command_space(
    robot_kinematics: str | None,
    algo_config: dict[str, Any],
    *,
    robot_command_mode: str | None = None,
) -> str:
    """Resolve robot command-space metadata for the current run.

    Returns:
        str: Canonical command-space label.
    """
    kin = str(robot_kinematics or _DEFAULT_KINEMATICS).strip().lower()
    if kin in {"holonomic", "omni", "omnidirectional"}:
        mode_source = (
            robot_command_mode
            if robot_command_mode is not None
            else algo_config.get("command_mode", "vx_vy")
        )
        mode = str(mode_source).strip().lower()
        return "holonomic_vxy_world" if mode == "vx_vy" else "unicycle_vw"
    return "unicycle_vw"


def init_feasibility_metadata(meta: dict[str, Any]) -> None:
    """Initialize mutable kinematics-feasibility counters in algorithm metadata."""
    meta["kinematics_feasibility"] = {
        "commands_evaluated": 0,
        "infeasible_native_count": 0,
        "projected_count": 0,
        "_sum_abs_delta_linear": 0.0,
        "_sum_abs_delta_angular": 0.0,
        "_max_abs_delta_linear": 0.0,
        "_max_abs_delta_angular": 0.0,
    }


def project_with_feasibility(
    *,
    model: KinematicsModel,
    command: tuple[float, float],
    meta: dict[str, Any],
) -> tuple[float, float]:
    """Project a command while accumulating feasibility diagnostics.

    Returns:
        tuple[float, float]: Projected command.
    """
    projected = model.project(command)
    feasibility = meta.get("kinematics_feasibility")
    if not isinstance(feasibility, dict):
        return projected
    feasible_native = bool(model.is_feasible(command))
    delta_linear = abs(float(projected[0]) - float(command[0]))
    delta_angular = abs(float(projected[1]) - float(command[1]))
    feasibility["commands_evaluated"] = int(feasibility.get("commands_evaluated", 0)) + 1
    if not feasible_native:
        feasibility["infeasible_native_count"] = (
            int(feasibility.get("infeasible_native_count", 0)) + 1
        )
    if command != projected:
        feasibility["projected_count"] = int(feasibility.get("projected_count", 0)) + 1
    feasibility["_sum_abs_delta_linear"] = float(
        feasibility.get("_sum_abs_delta_linear", 0.0)
    ) + float(delta_linear)
    feasibility["_sum_abs_delta_angular"] = float(
        feasibility.get("_sum_abs_delta_angular", 0.0)
    ) + float(delta_angular)
    feasibility["_max_abs_delta_linear"] = max(
        float(feasibility.get("_max_abs_delta_linear", 0.0)),
        float(delta_linear),
    )
    feasibility["_max_abs_delta_angular"] = max(
        float(feasibility.get("_max_abs_delta_angular", 0.0)),
        float(delta_angular),
    )
    return projected


def planner_kinematics_compatibility(
    *,
    algo: str,
    robot_kinematics: str,
    algo_config: dict[str, Any],
) -> tuple[bool, str | None]:
    """Return explicit compatibility status for planner/kinematics combinations."""
    algo_key = algo.strip().lower()
    kin = robot_kinematics.strip().lower()
    if kin in {"holonomic", "omni", "omnidirectional"} and algo_key == "rvo":
        return (
            False,
            f"planner '{algo_key}' is a placeholder adapter and is disabled for '{kin}' runs",
        )
    if kin in {"holonomic", "omni", "omnidirectional"} and algo_key == "dwa":
        return (
            False,
            "planner 'dwa' produces unicycle commands and is disabled for holonomic runs",
        )
    if algo_key == "ppo" and kin in {"holonomic", "omni", "omnidirectional"}:
        obs_mode = str(algo_config.get("obs_mode", "vector")).strip().lower()
        if obs_mode == "image":
            return (
                False,
                "ppo holonomic runs require non-image obs_mode for map-runner compatibility",
            )
    return True, None


def validate_planner_contract(
    *,
    algo: str,
    robot_kinematics: str,
    algo_config: dict[str, Any],
    observation_mode: str | None = None,
    observation_level: str | None = None,
) -> dict[str, Any]:
    """Validate planner observation/action compatibility before benchmark execution.

    Returns:
        dict[str, Any]: Serialized planner contract metadata for the requested planner.

    Raises:
        PlannerContractValidationError: If the planner is incompatible with the requested
        observation mode, robot kinematics, or action contract.
    """
    algo_key = algo.strip().lower()
    try:
        contract = planner_contract_for_algorithm(
            algo_key,
            observation_mode=observation_mode,
            observation_level=observation_level,
            robot_kinematics=robot_kinematics,
        )
    except ValueError as exc:
        raise PlannerContractValidationError(
            f"Planner contract mismatch for planner '{algo_key}': {exc}"
        ) from exc
    if (
        algo_key == "safety_barrier"
        and contract.observation_contract.active_mode == "sensor_fusion_state"
        and not algo_config.get("lidar_occupancy_adapter")
    ):
        raise PlannerContractValidationError(
            "Planner contract mismatch for planner 'safety_barrier': "
            "sensor_fusion_state/lidar_2d requires explicit "
            "algo_config['lidar_occupancy_adapter']."
        )

    payload = contract.to_metadata()
    compatible, reason = planner_kinematics_compatibility(
        algo=algo_key,
        robot_kinematics=robot_kinematics,
        algo_config=algo_config,
    )
    if not compatible:
        action = payload["action_contract"]
        raise PlannerContractValidationError(
            "Planner contract mismatch for "
            f"planner '{algo_key}' with robot_kinematics='{robot_kinematics}': {reason}. "
            f"Declared command_space='{action['command_space']}'."
        )
    return payload


__all__ = [
    "PlannerContractValidationError",
    "default_robot_command_space",
    "init_feasibility_metadata",
    "planner_kinematics_compatibility",
    "project_with_feasibility",
    "validate_planner_contract",
]
