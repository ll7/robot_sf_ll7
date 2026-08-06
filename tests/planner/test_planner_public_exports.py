"""Focused export contract for the robot_sf.planner package public surfaces.

Guards the reviewed ``__all__`` surface for issue #6800: every declared export
resolves to the pre-change object on its pre-change import path, no declared
name is missing, and missing, misspelled, or stale names never leak into the
public surface. Follows the pattern established in
``tests/sim/test_sim_public_exports.py`` for issue #6486.
"""

from __future__ import annotations

import importlib

import pytest

import robot_sf.planner as planner_facade

# Module -> reviewed public export list (ruff/isort-sorted within each module).
# Re-exported socnav base names are included because they are consumed through
# the family modules (e.g. ``socnav_orca.SamplingPlannerAdapter``).
PLANNER_MODULES_ALL = {
    "classic_planner_adapter": [
        "PlannerActionAdapter",
        "attach_classic_global_planner",
    ],
    "constants": [
        "DEFAULT_GMM_MODE_COUNT",
        "DEFAULT_STREAM_GAP_COMMIT_HOLD_STEPS",
        "DEFAULT_STREAM_GAP_FORWARD_LOOKAHEAD_M",
        "DEFAULT_STREAM_GAP_MAX_ANGULAR_SPEED",
        "DEFAULT_STREAM_GAP_MAX_LINEAR_SPEED",
        "DEFAULT_STREAM_GAP_SAMPLE_HORIZON_S",
    ],
    "dwa": ["DWAPlannerAdapter", "DWAPlannerConfig", "build_dwa_config"],
    "hybrid_global_rl": [
        "GridRouteWaypointProvider",
        "HybridGlobalRLLocalAdapter",
        "HybridGlobalRLLocalConfig",
        "WaypointDecision",
        "WaypointProvider",
        "build_hybrid_global_rl_config",
    ],
    "hybrid_route_corridor": [
        "lateral_offset_to_segment",
        "route_float",
        "route_point",
        "route_progress_pair",
        "route_tangent_heading",
    ],
    "kinematics_model": [
        "BicycleDriveKinematicsModel",
        "Command2D",
        "DifferentialDriveKinematicsModel",
        "HolonomicPassthroughKinematicsModel",
        "KinematicsModel",
        "resolve_benchmark_kinematics_model",
    ],
    "learned_prediction_mpc": [
        "LEARNED_PREDICTION_MPC_ALIASES",
        "build_learned_prediction_mpc_adapter",
    ],
    "learned_short_horizon_predictor": [
        "LearnedShortHorizonPedestrianPredictor",
        "LearnedShortHorizonPredictorConfig",
        "build_learned_short_horizon_predictor_config",
        "build_predictor_module",
        "encode_predictor_features",
        "pedestrian_world_state",
        "predictor_io_dims",
    ],
    "learned_short_horizon_trainer": [
        "CLAIM_BOUNDARY",
        "EVIDENCE_TIER",
        "REAL_TRAJECTORY_EVIDENCE_TIER",
        "SCHEMA_VERSION",
        "ShortHorizonTrainerConfig",
        "TrainingResult",
        "generate_real_trajectory_training_batch",
        "generate_synthetic_training_batch",
        "generate_training_batch",
        "train_short_horizon_predictor",
    ],
    "nmpc_social": [
        "NMPCSocialConfig",
        "NMPCSocialPlannerAdapter",
        "NMPCSolveResult",
        "build_nmpc_social_config",
    ],
    "obstacle_features": [
        "PREDICTIVE_EGO_FEATURE_DIM",
        "PREDICTIVE_EGO_FEATURE_SCHEMA",
        "PREDICTIVE_EGO_MOTION_CHANNEL_SLOTS",
        "PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME",
        "PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE",
        "PREDICTIVE_LEGACY_FEATURE_DIM",
        "PREDICTIVE_LEGACY_FEATURE_SCHEMA",
        "PREDICTIVE_OBSTACLE_FEATURE_DIM",
        "PREDICTIVE_OBSTACLE_FEATURE_SCHEMA",
        "PREDICTIVE_OBSTACLE_UNAVAILABLE_FEATURE_ROW",
        "LocalObstacleFeature",
        "LocalObstacleFeatureExtractor",
        "ObstacleFeatureSchema",
        "ObstacleFeatureSchemaError",
        "append_obstacle_features",
        "infer_predictive_feature_schema",
        "normalize_obstacle_lines",
        "obstacle_lines_from_map",
        "obstacle_lines_from_observation",
        "predictive_ego_motion_channel_producer_key",
        "predictive_feature_schema_metadata",
        "validate_predictive_feature_schema_metadata",
        "validate_predictive_runtime_feature_schema",
    ],
    "ompl_geometric_adapter": [
        "OmplGeometricAdapter",
        "OmplGeometricConfig",
        "OmplGeometricResult",
        "OmplPlannerChoice",
    ],
    "ompl_smoke": [
        "OmplSmokeConfig",
        "OmplSmokeResult",
        "check_ompl_available",
        "compare_with_classic_route",
        "smoke_plan",
    ],
    "path_smoother": ["douglas_peucker"],
    "poi_sampler": ["POISampler"],
    "prediction_mpc": [
        "ConstantVelocityPedestrianPredictor",
        "NullPedestrianPredictor",
        "PedestrianFuturePredictor",
        "PredictedPedestrianFutures",
        "PredictionMPCConfig",
        "PredictionMPCPlannerAdapter",
        "build_prediction_mpc_config",
    ],
    "sipp_lattice": [
        "COMPARISON_CONSISTENT_FEASIBLE",
        "COMPARISON_CONSISTENT_NOT_FEASIBLE",
        "COMPARISON_DIVERGENT_EXPLAINED",
        "COMPARISON_DIVERGENT_UNEXPECTED",
        "COMPARISON_INDETERMINATE",
        "EPISODE_LOCAL_POLICY_FAILURE",
        "EPISODE_NOT_PROVEN_FEASIBLE",
        "EPISODE_SUCCEEDED",
        "FEASIBILITY_FEASIBLE",
        "FEASIBILITY_NOT_PROVEN_FEASIBLE",
        "SPACE_TIME_FEASIBILITY_CLAIM_BOUNDARY",
        "SPACE_TIME_FEASIBILITY_ISSUE",
        "SPACE_TIME_FEASIBILITY_REVIEW_MARKER",
        "SPACE_TIME_FEASIBILITY_SCHEMA",
        "MotionPrimitive",
        "PedestrianOccupancyForecast",
        "PrimitiveKind",
        "SippKinodynamicCollisionModel",
        "SippLatticeConfig",
        "SippLatticePlannerAdapter",
        "SippLatticePrimitiveSet",
        "SippLatticeSearch",
        "SippLatticeSearchPlannerAdapter",
        "SippSearchResult",
        "SpaceTimeDiscretization",
        "SpaceTimeFeasibilityOracle",
        "SpaceTimeFeasibilityResult",
        "build_pedestrian_occupancy_forecast",
        "build_sipp_lattice_config",
        "build_sipp_lattice_search_adapter",
        "build_space_time_feasibility_oracle",
        "build_space_time_feasibility_oracle_from_algo_config",
        "classify_episode_feasibility",
        "compare_with_static_feasibility",
        "space_time_feasibility_result_to_dict",
    ],
    "socnav_base": [
        "SamplingPlannerAdapter",
        "SocNavBenchComplexPolicy",
        "SocNavPlannerConfig",
        "SocNavPlannerPolicy",
        "TrivialReferencePlannerAdapter",
    ],
    "socnav_occupancy": ["OccupancyAwarePlannerMixin"],
    "socnav_orca": [
        "HRVOPlannerAdapter",
        "ORCAPlannerAdapter",
        "SamplingPlannerAdapter",
        "SocNavPlannerConfig",
        "SocNavPlannerPolicy",
        "make_hrvo_policy",
        "make_orca_policy",
    ],
    "socnav_prediction": [
        "PredictionPlannerAdapter",
        "PredictiveTrajectoryModel",
        "SamplingPlannerAdapter",
        "SocNavBenchSamplingAdapter",
        "SocNavPlannerConfig",
        "SocNavPlannerPolicy",
        "make_prediction_policy",
    ],
    "socnav_sacadrl": [
        "SACADRLPlannerAdapter",
        "SamplingPlannerAdapter",
        "SocNavPlannerConfig",
        "SocNavPlannerPolicy",
        "make_sacadrl_policy",
    ],
    "socnav_social_force": [
        "SamplingPlannerAdapter",
        "SocNavPlannerConfig",
        "SocNavPlannerPolicy",
        "SocialForcePlannerAdapter",
        "make_social_force_policy",
        "sf_forces",
    ],
    "topology_parallel_nmpc": [
        "HypothesisDiagnostics",
        "TopologyParallelNMPCConfig",
        "TopologyParallelNMPCPlannerAdapter",
        "build_topology_parallel_nmpc_config",
    ],
    "visibility_graph": ["VisibilityGraph"],
    "visibility_planner": ["PlannerConfig", "PlanningFailedError", "VisibilityPlanner"],
    "visualization": ["ObstacleList", "plot_global_plan", "plot_visibility_graph"],
}

# Misspelled or stale names that must not appear in any module export list.
STALE_NAMES = [
    "PlannerActionAdaptere",  # misspelled adapter
    "DouglassPeucker",  # misspelled path smoother
    "SippLatticeSearchPlanner",  # stale pre-rename adapter name
    "VisibilityPlannerAdapter",  # stale wrapper name
    "_SACADRLModel",  # intentionally private lazy model helper
    "_sacadrl_actions",  # intentionally private lazy helper
]


def _module_path(module_name: str) -> str:
    return f"robot_sf.planner.{module_name}"


def _load_module(module_name: str):
    return importlib.import_module(_module_path(module_name))


@pytest.mark.parametrize("module_name", sorted(PLANNER_MODULES_ALL))
def test_planner_module_declares_reviewed_public_surface(module_name: str) -> None:
    """Each module declares exactly its reviewed ``__all__`` surface."""
    module = _load_module(module_name)
    assert module.__all__ == PLANNER_MODULES_ALL[module_name]
    assert set(module.__all__) <= set(dir(module))


@pytest.mark.parametrize(
    ("module_name", "name"),
    [(module_name, name) for module_name, names in PLANNER_MODULES_ALL.items() for name in names],
)
def test_planner_export_resolves_on_pre_change_path(module_name: str, name: str) -> None:
    """Every declared export resolves to the pre-change object.

    For classes and functions defined by the module itself, the pre-change
    identity includes the original qualname; re-exported aliases and type
    aliases intentionally point at their defining module and are only checked
    for resolvability.
    """
    module = _load_module(module_name)
    export = getattr(module, name)
    export_module = getattr(export, "__module__", None)
    export_qualname = getattr(export, "__qualname__", None)
    if export_module == _module_path(module_name) and export_qualname is not None:
        assert export_qualname == name


@pytest.mark.parametrize("name", STALE_NAMES)
def test_planner_stale_and_private_names_are_not_exported(name: str) -> None:
    """Misspelled, stale, and private names stay out of every export list."""
    for declared in PLANNER_MODULES_ALL.values():
        assert name not in declared


@pytest.mark.parametrize(
    "name",
    [
        "GlobalPlanner",  # facade-only alias, not a module export
        "BaselineStepToLocalAdapter",  # facade export from planner.protocol
    ],
)
def test_planner_facade_export_is_not_a_module_export(name: str) -> None:
    """Facade exports that do not belong to any reviewed module stay out."""
    for declared in PLANNER_MODULES_ALL.values():
        assert name not in declared


def test_planner_facade_exports_resolve_through_lazy_machinery() -> None:
    """Declared facade exports still resolve on their pre-change paths."""
    for name in planner_facade.__all__:
        export = getattr(planner_facade, name)
        assert export is not None
