"""Global planner package for SVG-based waypoint generation.

Public planner exports are resolved lazily so importing lightweight navigation or
environment modules does not pull optional learned-risk/training dependencies
such as PyTorch into core wheel installs.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "LocalPlannerProtocol": "robot_sf.planner.protocol",
    "BaselineStepToLocalAdapter": "robot_sf.planner.protocol",
    "BaselineActionProjector": "robot_sf.planner.protocol",
    "BaselineStepExecutor": "robot_sf.planner.protocol",
    "normalize_planner_diagnostics": "robot_sf.planner.protocol",
    "ClassicGlobalPlanner": "robot_sf.planner.classic_global_planner",
    "ClassicPlannerConfig": "robot_sf.planner.classic_global_planner",
    "DWAPlannerAdapter": "robot_sf.planner.dwa",
    "DWAPlannerConfig": "robot_sf.planner.dwa",
    "AnisotropicGaussianCostConfig": "robot_sf.planner.anisotropic_gaussian_cost",
    "AnisotropicGaussianCostPlanner": "robot_sf.planner.anisotropic_gaussian_cost",
    "build_anisotropic_gaussian_cost_config": ("robot_sf.planner.anisotropic_gaussian_cost"),
    "evaluate_anisotropic_gaussian_cost": ("robot_sf.planner.anisotropic_gaussian_cost"),
    "evaluate_anisotropic_repulsive_force": ("robot_sf.planner.anisotropic_gaussian_cost"),
    "ForceCoupledPotentialFieldConfig": "robot_sf.planner.force_coupled_potential_field",
    "ForceCoupledPotentialFieldPlanner": "robot_sf.planner.force_coupled_potential_field",
    "build_force_coupled_potential_field_config": (
        "robot_sf.planner.force_coupled_potential_field"
    ),
    "PredictiveGaussianHumanCost": "robot_sf.planner.predictive_human_cost",
    "PredictiveGaussianHumanCostConfig": "robot_sf.planner.predictive_human_cost",
    "build_predictive_gaussian_human_cost_config": "robot_sf.planner.predictive_human_cost",
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
    "SippLatticeConfig": "robot_sf.planner.sipp_lattice",
    "SippLatticePlannerAdapter": "robot_sf.planner.sipp_lattice",
    "SippLatticeSearchPlannerAdapter": "robot_sf.planner.sipp_lattice",
    "build_sipp_lattice_config": "robot_sf.planner.sipp_lattice",
    "build_sipp_lattice_search_adapter": "robot_sf.planner.sipp_lattice",
    "TEBCommitmentConfig": "robot_sf.planner.teb_commitment",
    "TEBCommitmentPlannerAdapter": "robot_sf.planner.teb_commitment",
    "PlannerConfig": "robot_sf.planner.visibility_planner",
    "PlanningFailedError": "robot_sf.planner.visibility_planner",
    "VisibilityPlanner": "robot_sf.planner.visibility_planner",
    "plot_global_plan": "robot_sf.planner.visualization",
    "plot_visibility_graph": "robot_sf.planner.visualization",
    "TopologyParallelNMPCPlannerAdapter": "robot_sf.planner.topology_parallel_nmpc",
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
