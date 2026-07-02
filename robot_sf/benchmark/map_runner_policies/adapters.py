"""Builders for adapter-backed map-runner policy families.

Continuation of the ``_build_policy`` decomposition (#3384), building on the package
registry (#3400) and the neutral ``build_adapter_policy`` helper (#3403). Each builder
is a faithful move of the corresponding ``if algo_key in {...}`` branch from
``robot_sf.benchmark.map_runner`` — no semantic change.

This module migrates the two adapter families whose dependencies all resolve from
lower-level ``robot_sf.planner`` modules (no ``map_runner``-local helpers), so the
import graph stays acyclic. Families that still depend on ``map_runner``-local
helpers (e.g. ``trivial_reference`` via ``_build_socnav_config``) are deferred to a
later slice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.map_runner_policy_common import build_adapter_policy
from robot_sf.planner.learned_risk_surface import (
    RiskSurfacePlannerAdapter,
    build_local_risk_surface_spec,
)
from robot_sf.planner.lidar_tracked_agents import build_lidar_tracked_social_force_adapter
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, build_risk_dwa_config

if TYPE_CHECKING:
    from collections.abc import Callable

#: Algorithm keys handled by the risk-surface DWA builder.
RISK_SURFACE_DWA_KEYS = frozenset({"risk_surface_dwa", "risk_surface_dwa_v0"})
#: Algorithm keys handled by the lidar tracked social-force builder.
LIDAR_SOCIAL_FORCE_KEYS = frozenset({"lidar_social_force", "lidar_tracked_social_force"})


def build_risk_surface_dwa(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build the deterministic risk-surface DWA adapter policy.

    Returns:
        Policy callable and enriched metadata dictionary.
    """
    del adapter_impact_eval
    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    risk_surface_cfg = (
        algo_config.get("risk_surface") if isinstance(algo_config.get("risk_surface"), dict) else {}
    )
    risk_dwa_cfg = (
        algo_config.get("risk_dwa") if isinstance(algo_config.get("risk_dwa"), dict) else {}
    )
    adapter = RiskSurfacePlannerAdapter(
        spec=build_local_risk_surface_spec(risk_surface_cfg),
        planner=RiskDWAPlannerAdapter(config=build_risk_dwa_config(risk_dwa_cfg)),
    )
    meta: dict[str, Any] = {"algorithm": algo_key}
    meta["risk_surface_planner"] = {
        "status": "enabled",
        "producer": "deterministic_pedestrian_risk_surface",
        "wrapped_planner": "risk_dwa",
        "benchmark_strength": False,
        "claim_boundary": "exploratory_smoke_only",
    }
    return build_adapter_policy(
        algo_key="risk_surface_dwa",
        algo_config=algo_config,
        meta=meta,
        adapter=adapter,
        adapter_name="RiskSurfacePlannerAdapter",
        robot_kinematics=robot_kinematics,
        normalized_robot_command_mode=normalized_robot_command_mode,
        limitations="deterministic_risk_surface_fixture_not_benchmark_evidence",
    )


def build_lidar_social_force(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build the lidar endpoint-tracked social-force adapter policy.

    Returns:
        Policy callable and enriched metadata dictionary.
    """
    del adapter_impact_eval
    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    adapter = build_lidar_tracked_social_force_adapter(algo_config)
    meta: dict[str, Any] = {"algorithm": algo_key}
    return build_adapter_policy(
        algo_key="lidar_social_force",
        algo_config=algo_config,
        meta=meta,
        adapter=adapter,
        adapter_name="LidarTrackedSocialForceAdapter",
        robot_kinematics=robot_kinematics,
        normalized_robot_command_mode=normalized_robot_command_mode,
        limitations="lidar_endpoint_tracked_social_force_testing_only",
    )
