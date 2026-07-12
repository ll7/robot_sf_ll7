"""Global planner package for SVG-based waypoint generation.

Public planner exports are resolved lazily so importing lightweight navigation or
environment modules does not pull optional learned-risk/training dependencies
such as PyTorch into core wheel installs.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ClassicGlobalPlanner": "robot_sf.planner.classic_global_planner",
    "ClassicPlannerConfig": "robot_sf.planner.classic_global_planner",
    "DWAPlannerAdapter": "robot_sf.planner.dwa",
    "DWAPlannerConfig": "robot_sf.planner.dwa",
    "PlanningError": "robot_sf.planner.classic_global_planner",
    "PlannerActionAdapter": "robot_sf.planner.classic_planner_adapter",
    "attach_classic_global_planner": "robot_sf.planner.classic_planner_adapter",
    "LocalRiskSurface": "robot_sf.planner.learned_risk_surface",
    "LocalRiskSurfaceSpec": "robot_sf.planner.learned_risk_surface",
    "RiskSurfacePlannerAdapter": "robot_sf.planner.learned_risk_surface",
    "RiskSurfaceUnavailable": "robot_sf.planner.learned_risk_surface",
    "attach_risk_surface_to_observation": "robot_sf.planner.learned_risk_surface",
    "build_local_risk_surface_spec": "robot_sf.planner.learned_risk_surface",
    "deterministic_pedestrian_risk_surface": "robot_sf.planner.learned_risk_surface",
    "POISampler": "robot_sf.planner.poi_sampler",
    "PolicyStackV1Adapter": "robot_sf.planner.policy_stack_v1",
    "PolicyStackV1Config": "robot_sf.planner.policy_stack_v1",
    "RiskDWAPlannerAdapter": "robot_sf.planner.risk_dwa",
    "RiskDWAPlannerConfig": "robot_sf.planner.risk_dwa",
    "TEBCommitmentConfig": "robot_sf.planner.teb_commitment",
    "TEBCommitmentPlannerAdapter": "robot_sf.planner.teb_commitment",
    "TopologyParallelNMPCConfig": "robot_sf.planner.topology_parallel_nmpc",
    "TopologyParallelNMPCPlannerAdapter": "robot_sf.planner.topology_parallel_nmpc",
    "build_topology_parallel_nmpc_config": "robot_sf.planner.topology_parallel_nmpc",
    "NMPCSolveResult": "robot_sf.planner.nmpc_social",
    "PlannerConfig": "robot_sf.planner.visibility_planner",
    "PlanningFailedError": "robot_sf.planner.visibility_planner",
    "VisibilityPlanner": "robot_sf.planner.visibility_planner",
    "plot_global_plan": "robot_sf.planner.visualization",
    "plot_visibility_graph": "robot_sf.planner.visualization",
}

__all__ = sorted([*_LAZY_EXPORTS, "GlobalPlanner"])  # noqa: PLE0605


def __getattr__(name: str) -> Any:
    """Resolve planner exports on first access.

    Returns:
        Exported planner class, function, or compatibility alias.
    """

    if name == "GlobalPlanner":
        value = __getattr__("VisibilityPlanner")
        globals()[name] = value
        return value
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
