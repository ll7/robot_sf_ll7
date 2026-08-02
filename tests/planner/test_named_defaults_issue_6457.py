"""Regression coverage for the named planner defaults from issue #6457."""

from robot_sf.planner.adaptive_proxemic_selector import (
    _DEFAULT_HIGH_DENSITY_COUNT,
    build_adaptive_proxemic_selector_config,
)
from robot_sf.planner.constants import DEFAULT_STREAM_GAP_COMMIT_HOLD_STEPS
from robot_sf.planner.gap_prediction import build_gap_prediction_config
from robot_sf.planner.grid_route import (
    _DEFAULT_CLEARANCE_SEARCH_CELLS,
    _DEFAULT_WAYPOINT_LOOKAHEAD_CELLS,
    GridRoutePlannerConfig,
    build_grid_route_config,
)
from robot_sf.planner.guarded_ppo import (
    _DEFAULT_GUARD_ROLLOUT_STEPS,
    GuardedPPOConfig,
    build_guarded_ppo_config,
)
from robot_sf.planner.hybrid_global_rl import (
    _DEFAULT_WAYPOINT_MAX_DISTANCE_FROM_ROBOT_M,
    HybridGlobalRLLocalConfig,
    build_hybrid_global_rl_config,
)
from robot_sf.planner.hybrid_orca_sampler import (
    _DEFAULT_ROLLOUT_STEPS,
    _DEFAULT_ROUTE_STALL_CYCLES_BEFORE_SAMPLER,
    HybridORCASamplerConfig,
    build_hybrid_orca_sampler_build_config,
)
from robot_sf.planner.hybrid_portfolio import (
    _DEFAULT_DENSE_PED_COUNT as PORTFOLIO_DENSE_PED_COUNT,
)
from robot_sf.planner.hybrid_portfolio import (
    _DEFAULT_HYSTERESIS_STEPS,
    HybridPortfolioConfig,
    build_hybrid_portfolio_build_config,
)
from robot_sf.planner.mppi_social import (
    _DEFAULT_ITERATIONS as MPPI_SOCIAL_ITERATIONS,
)
from robot_sf.planner.mppi_social import (
    MPPISocialConfig,
    build_mppi_social_config,
)
from robot_sf.planner.planner_selector_v2_diagnostic import (
    _DEFAULT_DENSE_PED_COUNT as SELECTOR_DENSE_PED_COUNT,
)
from robot_sf.planner.planner_selector_v2_diagnostic import (
    PlannerSelectorV2DiagnosticConfig,
    build_planner_selector_v2_diagnostic_config,
)
from robot_sf.planner.predictive_mppi import (
    _DEFAULT_CLEARANCE_WEIGHT,
    _DEFAULT_GOAL_PROGRESS_WEIGHT,
    _DEFAULT_PROGRESS_ESCAPE_DISTANCE_M,
    PredictiveMPPIConfig,
    build_predictive_mppi_config,
)
from robot_sf.planner.predictive_mppi import (
    _DEFAULT_ITERATIONS as PREDICTIVE_MPPI_ITERATIONS,
)
from robot_sf.planner.risk_dwa import (
    _DEFAULT_OBSTACLE_CLEARANCE_WEIGHT,
    RiskDWAPlannerConfig,
    build_risk_dwa_config,
)
from robot_sf.planner.socnav_prediction import _DEFAULT_FORECAST_VARIANT_RISK_DISTANCE_M
from robot_sf.planner.stream_gap import StreamGapPlannerConfig
from robot_sf.planner.topology_guided_local_policy import (
    _DEFAULT_BLOCK_RADIUS_CELLS,
    _DEFAULT_PRIMARY_ROUTE_REUSE_PENALTY_COOLDOWN_STEPS,
    _DEFAULT_ROUTE_GUIDE_OBSTACLE_INFLATION_CELLS,
    TopologyGuidedLocalPolicyConfig,
    build_topology_guided_local_policy_config,
)


def test_issue_6457_named_default_values_preserve_builder_contracts() -> None:
    """Named defaults retain the exact values used by planner config builders."""
    adaptive = build_adaptive_proxemic_selector_config({})
    gap = build_gap_prediction_config({}).stream_gap
    grid_route = build_grid_route_config({})
    guarded_ppo = build_guarded_ppo_config({})
    hybrid_global = build_hybrid_global_rl_config({})
    hybrid_orca = build_hybrid_orca_sampler_build_config({}).guard
    hybrid_portfolio = build_hybrid_portfolio_build_config({}).hybrid
    mppi_social = build_mppi_social_config({})
    selector = build_planner_selector_v2_diagnostic_config({}).selector
    predictive_mppi = build_predictive_mppi_config({})
    risk_dwa = build_risk_dwa_config({})
    topology = build_topology_guided_local_policy_config({})

    assert _DEFAULT_HIGH_DENSITY_COUNT == adaptive.high_density_count == 3
    assert DEFAULT_STREAM_GAP_COMMIT_HOLD_STEPS == StreamGapPlannerConfig().commit_hold_steps == 6
    assert DEFAULT_STREAM_GAP_COMMIT_HOLD_STEPS == gap.commit_hold_steps == 6
    assert _DEFAULT_WAYPOINT_LOOKAHEAD_CELLS == grid_route.waypoint_lookahead_cells == 5
    assert _DEFAULT_CLEARANCE_SEARCH_CELLS == grid_route.clearance_search_cells == 5
    assert _DEFAULT_GUARD_ROLLOUT_STEPS == guarded_ppo.rollout_steps == 6
    assert (
        _DEFAULT_WAYPOINT_MAX_DISTANCE_FROM_ROBOT_M
        == hybrid_global.waypoint_max_distance_from_robot
        == 3.0
    )
    assert _DEFAULT_ROLLOUT_STEPS == hybrid_orca.rollout_steps == 6
    assert (
        _DEFAULT_ROUTE_STALL_CYCLES_BEFORE_SAMPLER
        == hybrid_orca.route_stall_cycles_before_sampler
        == 3
    )
    assert PORTFOLIO_DENSE_PED_COUNT == hybrid_portfolio.dense_ped_count == 6
    assert _DEFAULT_HYSTERESIS_STEPS == hybrid_portfolio.hysteresis_steps == 6
    assert MPPI_SOCIAL_ITERATIONS == mppi_social.iterations == 3
    assert SELECTOR_DENSE_PED_COUNT == selector.dense_ped_count == 4
    assert PREDICTIVE_MPPI_ITERATIONS == predictive_mppi.iterations == 4
    assert _DEFAULT_GOAL_PROGRESS_WEIGHT == predictive_mppi.goal_progress_weight == 6.0
    assert _DEFAULT_CLEARANCE_WEIGHT == predictive_mppi.clearance_weight == 3.0
    assert _DEFAULT_PROGRESS_ESCAPE_DISTANCE_M == predictive_mppi.progress_escape_distance == 1.2
    assert _DEFAULT_OBSTACLE_CLEARANCE_WEIGHT == risk_dwa.obstacle_clearance_weight == 1.2
    assert _DEFAULT_FORECAST_VARIANT_RISK_DISTANCE_M == 3.0
    assert (
        _DEFAULT_ROUTE_GUIDE_OBSTACLE_INFLATION_CELLS
        == topology.route_hypothesis.obstacle_inflation_cells
        == 3
    )
    assert _DEFAULT_BLOCK_RADIUS_CELLS == topology.block_radius_cells == 3
    assert (
        _DEFAULT_PRIMARY_ROUTE_REUSE_PENALTY_COOLDOWN_STEPS
        == topology.primary_route_reuse_penalty_cooldown_steps
        == 3
    )


def test_issue_6457_named_defaults_cover_direct_dataclass_construction() -> None:
    """Direct config construction uses the same named defaults as each builder."""
    predictive = build_predictive_mppi_config({})
    topology = build_topology_guided_local_policy_config({})

    assert GridRoutePlannerConfig().waypoint_lookahead_cells == _DEFAULT_WAYPOINT_LOOKAHEAD_CELLS
    assert GridRoutePlannerConfig().clearance_search_cells == _DEFAULT_CLEARANCE_SEARCH_CELLS
    assert GuardedPPOConfig().rollout_steps == _DEFAULT_GUARD_ROLLOUT_STEPS
    assert (
        HybridGlobalRLLocalConfig().waypoint_max_distance_from_robot
        == _DEFAULT_WAYPOINT_MAX_DISTANCE_FROM_ROBOT_M
    )
    assert HybridORCASamplerConfig().rollout_steps == _DEFAULT_ROLLOUT_STEPS
    assert (
        HybridORCASamplerConfig().route_stall_cycles_before_sampler
        == _DEFAULT_ROUTE_STALL_CYCLES_BEFORE_SAMPLER
    )
    assert HybridPortfolioConfig().dense_ped_count == PORTFOLIO_DENSE_PED_COUNT
    assert HybridPortfolioConfig().hysteresis_steps == _DEFAULT_HYSTERESIS_STEPS
    assert MPPISocialConfig().iterations == MPPI_SOCIAL_ITERATIONS
    assert PlannerSelectorV2DiagnosticConfig().dense_ped_count == SELECTOR_DENSE_PED_COUNT
    assert PredictiveMPPIConfig(socnav=predictive.socnav).iterations == PREDICTIVE_MPPI_ITERATIONS
    assert (
        PredictiveMPPIConfig(socnav=predictive.socnav).goal_progress_weight
        == _DEFAULT_GOAL_PROGRESS_WEIGHT
    )
    assert (
        PredictiveMPPIConfig(socnav=predictive.socnav).clearance_weight == _DEFAULT_CLEARANCE_WEIGHT
    )
    assert (
        PredictiveMPPIConfig(socnav=predictive.socnav).progress_escape_distance
        == _DEFAULT_PROGRESS_ESCAPE_DISTANCE_M
    )
    assert RiskDWAPlannerConfig().obstacle_clearance_weight == _DEFAULT_OBSTACLE_CLEARANCE_WEIGHT
    assert (
        TopologyGuidedLocalPolicyConfig(
            hybrid_rule=topology.hybrid_rule,
            route_hypothesis=topology.route_hypothesis,
        ).block_radius_cells
        == _DEFAULT_BLOCK_RADIUS_CELLS
    )
    assert (
        TopologyGuidedLocalPolicyConfig(
            hybrid_rule=topology.hybrid_rule,
            route_hypothesis=topology.route_hypothesis,
        ).primary_route_reuse_penalty_cooldown_steps
        == _DEFAULT_PRIMARY_ROUTE_REUSE_PENALTY_COOLDOWN_STEPS
    )
