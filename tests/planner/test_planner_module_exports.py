"""Export-contract guard for the remaining ``robot_sf.planner`` modules.

Split child of #6477 (repository-wide ``__all__`` sweep), following the
import-identity pattern established by #6486 / PR #6762 and #6797 / PR #6802.
Guards the reviewed ``__all__`` surfaces added by issue #6800:

- every declared export resolves to the pre-change object on its pre-change
  module path,
- no declared name is missing or misspelled,
- private, foreign, and stale names never leak into the public surface,
- the package facade stays lazy: importing it must not eagerly import optional
  dependencies such as OMPL or torch.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

CLASSIC_PLANNER_ADAPTER_ALL = [
    "PlannerActionAdapter",
    "attach_classic_global_planner",
]

CONSTANTS_ALL = [
    "DEFAULT_GMM_MODE_COUNT",
    "DEFAULT_STREAM_GAP_COMMIT_HOLD_STEPS",
    "DEFAULT_STREAM_GAP_FORWARD_LOOKAHEAD_M",
    "DEFAULT_STREAM_GAP_MAX_ANGULAR_SPEED",
    "DEFAULT_STREAM_GAP_MAX_LINEAR_SPEED",
    "DEFAULT_STREAM_GAP_SAMPLE_HORIZON_S",
]

DWA_ALL = [
    "DWAPlannerAdapter",
    "DWAPlannerConfig",
    "build_dwa_config",
]

HYBRID_GLOBAL_RL_ALL = [
    "GridRouteWaypointProvider",
    "HybridGlobalRLLocalAdapter",
    "HybridGlobalRLLocalConfig",
    "WaypointDecision",
    "WaypointProvider",
    "build_hybrid_global_rl_config",
]

HYBRID_ROUTE_CORRIDOR_ALL = [
    "lateral_offset_to_segment",
    "route_float",
    "route_point",
    "route_progress_pair",
    "route_tangent_heading",
]

KINEMATICS_MODEL_ALL = [
    "BicycleDriveKinematicsModel",
    "Command2D",
    "DifferentialDriveKinematicsModel",
    "HolonomicPassthroughKinematicsModel",
    "KinematicsModel",
    "resolve_benchmark_kinematics_model",
]

LEARNED_PREDICTION_MPC_ALL = [
    "LEARNED_PREDICTION_MPC_ALIASES",
    "build_learned_prediction_mpc_adapter",
]

LEARNED_SHORT_HORIZON_PREDICTOR_ALL = [
    "LearnedShortHorizonPedestrianPredictor",
    "LearnedShortHorizonPredictorConfig",
    "build_learned_short_horizon_predictor_config",
    "build_predictor_module",
    "encode_predictor_features",
    "pedestrian_world_state",
    "predictor_io_dims",
]

LEARNED_SHORT_HORIZON_TRAINER_ALL = [
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
]

NMPC_SOCIAL_ALL = [
    "NMPCSocialConfig",
    "NMPCSocialPlannerAdapter",
    "NMPCSolveResult",
    "build_nmpc_social_config",
]

OBSTACLE_FEATURES_ALL = [
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
]

OMPL_GEOMETRIC_ADAPTER_ALL = [
    "OmplGeometricAdapter",
    "OmplGeometricConfig",
    "OmplGeometricResult",
    "OmplPlannerChoice",
]

OMPL_SMOKE_ALL = [
    "OmplSmokeConfig",
    "OmplSmokeResult",
    "check_ompl_available",
    "compare_with_classic_route",
    "smoke_plan",
]

PATH_SMOOTHER_ALL = [
    "douglas_peucker",
]

POI_SAMPLER_ALL = [
    "POISampler",
]

PREDICTION_MPC_ALL = [
    "ConstantVelocityPedestrianPredictor",
    "NullPedestrianPredictor",
    "PedestrianFuturePredictor",
    "PredictedPedestrianFutures",
    "PredictionMPCConfig",
    "PredictionMPCPlannerAdapter",
    "build_prediction_mpc_config",
]

SIPP_LATTICE_ALL = [
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
]

SOCNAV_BASE_ALL = [
    "SamplingPlannerAdapter",
    "SocNavBenchComplexPolicy",
    "SocNavPlannerConfig",
    "SocNavPlannerPolicy",
    "TrivialReferencePlannerAdapter",
]

SOCNAV_OCCUPANCY_ALL = [
    "OccupancyAwarePlannerMixin",
]

SOCNAV_ORCA_ALL = [
    "HRVOPlannerAdapter",
    "ORCAPlannerAdapter",
    "make_hrvo_policy",
    "make_orca_policy",
]

SOCNAV_PREDICTION_ALL = [
    "PredictionPlannerAdapter",
    "SocNavBenchSamplingAdapter",
    "make_prediction_policy",
]

SOCNAV_SACADRL_ALL = [
    "SACADRLPlannerAdapter",
    "make_sacadrl_policy",
]

SOCNAV_SOCIAL_FORCE_ALL = [
    "SocialForcePlannerAdapter",
    "make_social_force_policy",
]

TOPOLOGY_PARALLEL_NMPC_ALL = [
    "HypothesisDiagnostics",
    "TopologyParallelNMPCConfig",
    "TopologyParallelNMPCPlannerAdapter",
    "build_topology_parallel_nmpc_config",
]

VISIBILITY_GRAPH_ALL = [
    "VisibilityGraph",
]

VISIBILITY_PLANNER_ALL = [
    "PlannerConfig",
    "PlanningFailedError",
    "VisibilityPlanner",
]

VISUALIZATION_ALL = [
    "ObstacleList",
    "plot_global_plan",
    "plot_visibility_graph",
]

# module_name -> (expected __all__, reviewed list)
REVIEWED_SURFACES: dict[str, list[str]] = {
    "robot_sf.planner.classic_planner_adapter": CLASSIC_PLANNER_ADAPTER_ALL,
    "robot_sf.planner.constants": CONSTANTS_ALL,
    "robot_sf.planner.dwa": DWA_ALL,
    "robot_sf.planner.hybrid_global_rl": HYBRID_GLOBAL_RL_ALL,
    "robot_sf.planner.hybrid_route_corridor": HYBRID_ROUTE_CORRIDOR_ALL,
    "robot_sf.planner.kinematics_model": KINEMATICS_MODEL_ALL,
    "robot_sf.planner.learned_prediction_mpc": LEARNED_PREDICTION_MPC_ALL,
    "robot_sf.planner.learned_short_horizon_predictor": LEARNED_SHORT_HORIZON_PREDICTOR_ALL,
    "robot_sf.planner.learned_short_horizon_trainer": LEARNED_SHORT_HORIZON_TRAINER_ALL,
    "robot_sf.planner.nmpc_social": NMPC_SOCIAL_ALL,
    "robot_sf.planner.obstacle_features": OBSTACLE_FEATURES_ALL,
    "robot_sf.planner.ompl_geometric_adapter": OMPL_GEOMETRIC_ADAPTER_ALL,
    "robot_sf.planner.ompl_smoke": OMPL_SMOKE_ALL,
    "robot_sf.planner.path_smoother": PATH_SMOOTHER_ALL,
    "robot_sf.planner.poi_sampler": POI_SAMPLER_ALL,
    "robot_sf.planner.prediction_mpc": PREDICTION_MPC_ALL,
    "robot_sf.planner.sipp_lattice": SIPP_LATTICE_ALL,
    "robot_sf.planner.socnav_base": SOCNAV_BASE_ALL,
    "robot_sf.planner.socnav_occupancy": SOCNAV_OCCUPANCY_ALL,
    "robot_sf.planner.socnav_orca": SOCNAV_ORCA_ALL,
    "robot_sf.planner.socnav_prediction": SOCNAV_PREDICTION_ALL,
    "robot_sf.planner.socnav_sacadrl": SOCNAV_SACADRL_ALL,
    "robot_sf.planner.socnav_social_force": SOCNAV_SOCIAL_FORCE_ALL,
    "robot_sf.planner.topology_parallel_nmpc": TOPOLOGY_PARALLEL_NMPC_ALL,
    "robot_sf.planner.visibility_graph": VISIBILITY_GRAPH_ALL,
    "robot_sf.planner.visibility_planner": VISIBILITY_PLANNER_ALL,
    "robot_sf.planner.visualization": VISUALIZATION_ALL,
}


@pytest.mark.parametrize("module_name", sorted(REVIEWED_SURFACES))
def test_module_declares_the_reviewed_export_surface(module_name: str) -> None:
    """Each module exports exactly its reviewed ``__all__`` list."""
    module = importlib.import_module(module_name)
    assert module.__all__ == REVIEWED_SURFACES[module_name]
    assert set(module.__all__) <= set(dir(module))


@pytest.mark.parametrize(
    ("module_name", "name"),
    [(module_name, name) for module_name, names in REVIEWED_SURFACES.items() for name in names],
)
def test_module_exports_resolve_on_pre_change_paths(module_name: str, name: str) -> None:
    """Every declared export resolves to the pre-change module-level binding."""
    module = importlib.import_module(module_name)
    assert getattr(module, name) is module.__dict__[name]


@pytest.mark.parametrize(
    ("module_name", "name"),
    [
        ("robot_sf.planner.classic_planner_adapter", "_default_kinematics_model"),
        ("robot_sf.planner.dwa", "_reachable_interval"),
        ("robot_sf.planner.hybrid_global_rl", "_DEFAULT_WAYPOINT_MAX_DISTANCE_FROM_ROBOT_M"),
        ("robot_sf.planner.kinematics_model", "_build_diagnostics"),
        ("robot_sf.planner.learned_short_horizon_predictor", "_TinyMlp"),
        ("robot_sf.planner.nmpc_social", "_parse_bool"),
        ("robot_sf.planner.obstacle_features", "_coerce_point"),
        ("robot_sf.planner.ompl_geometric_adapter", "_build_obstacle_union"),
        ("robot_sf.planner.prediction_mpc", "_to_nmpc_config"),
        ("robot_sf.planner.sipp_lattice", "_SearchNode"),
        ("robot_sf.planner.socnav_base", "_SOCNAV_IMPORT_LOCK"),
        ("robot_sf.planner.topology_parallel_nmpc", "_material_separation"),
        ("robot_sf.planner.visibility_planner", "_dedup_consecutive"),
        ("robot_sf.planner.visualization", "_init_axes"),
    ],
)
def test_module_keeps_private_and_internal_names_unexported(module_name: str, name: str) -> None:
    """Private, foreign, and stale symbols stay out of every export list."""
    module = importlib.import_module(module_name)
    assert name not in module.__all__


def test_planner_facade_import_stays_lazy() -> None:
    """Importing the planner facade must not import optional deps eagerly.

    Guards the optional OMPL/torch/learned-risk dependency policy: ``import
    robot_sf.planner`` must stay cheap even when those optional dependencies are
    missing, and module-level ``__all__`` additions must not change that.
    """
    script = (
        "import sys\n"
        "import robot_sf.planner\n"
        "assert 'robot_sf.planner.ompl_geometric_adapter' not in sys.modules\n"
        "assert not any(key.startswith('ompl') for key in sys.modules)\n"
        "assert 'torch' not in sys.modules\n"
        "print('lazy-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "lazy-ok" in result.stdout
