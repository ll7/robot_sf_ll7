"""Characterization baseline for ``HybridRuleLocalPlannerAdapter``.

This file is the pre-decomposition behavior lock for
`robot_sf/planner/hybrid_rule_local_planner.py` (issue #4964, part of #4770).
It pins the observable output of the public API so the follow-up god-class
decomposition refactor (#4987) can prove behavior preservation by diffing
against these golden values.

Pinned surfaces (additive golden-value style, deterministic seed):
1. ``build_hybrid_rule_local_planner_config(...)`` for a fixed raw-config dict,
   including the nested ``proxemic_costmap`` expansion path.
2. ``HybridRuleLocalPlannerAdapter.plan(observation)`` exact ``(v, omega)``
   output for four fixed synthetic scenarios:
   - nominal route-following,
   - near-pedestrian speed-cap case,
   - static-clearance rejection case,
   - goal-posterior-active case.
3. The public ``diagnostics()`` / ``last_decision()`` dict shape (keys plus the
   rejection-count structure) after those plan calls.

Determinism is asserted explicitly across two consecutive planner instances in
``test_characterization_is_deterministic_across_instances``. This test does not
modify production code.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
    build_hybrid_rule_local_planner_config,
)

# ---------------------------------------------------------------------------
# Shared observation builders
#
# These mirror the compact payload shape used by the sibling
# ``test_hybrid_rule_local_planner.py`` so the baseline covers the same
# observation contract the rest of the suite exercises.
# ---------------------------------------------------------------------------


def _obs(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    goal: tuple[float, float] = (4.0, 0.0),
    speed: float = 0.0,
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> dict:
    """Build a compact observation payload for the hybrid-rule planner."""
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float).reshape(-1, 2),
            "velocities": np.asarray(ped_velocities, dtype=float).reshape(-1, 2),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": 0.25,
        },
        "sim": {"timestep": 0.1},
    }


def _obs_with_grid(
    *,
    goal: tuple[float, float] = (4.0, 0.0),
    occupied_cells: list[tuple[int, int]],
) -> dict:
    """Build an observation carrying a static-occupancy grid in front of the robot.

    Grid convention (matches ``test_hybrid_rule_local_planner.py``):
    - shape ``(channels=3, rows=21, cols=21)`` with resolution 0.2 m,
    - origin ``[-2.0, -2.0]`` so the robot at ``(0.0, 0.0)`` maps to cell
      ``(row=10, col=10)``,
    - ``channel_indices=[0, 1, 2]`` -> channel 0 = obstacles, 1 = pedestrians,
      2 = robot; obstacle cells are also mirrored to the robot channel to mark
      them occupied.
    """
    obs = _obs(goal=goal)
    grid = np.zeros((3, 21, 21), dtype=float)
    for row, col in occupied_cells:
        grid[0, row, col] = 1.0
        grid[2, row, col] = 1.0
    obs["occupancy_grid"] = grid
    obs["occupancy_grid_meta"] = {
        "origin": [-2.0, -2.0],
        "resolution": [0.2],
        "size": [4.2, 4.2],
        "use_ego_frame": [0.0],
        "center_on_robot": [0.0],
        "channel_indices": [0, 1, 2],
        "robot_pose": [0.0, 0.0, 0.0],
    }
    return obs


# ---------------------------------------------------------------------------
# Fixed scenario fixtures
#
# Every scenario is a module-level constant so the golden values below are tied
# to an unchanging input. Changing a fixture requires recomputing the golden
# values in the same commit.
# ---------------------------------------------------------------------------

# Scenario 1: open-space nominal route-following. No pedestrians, no obstacles.
_OBS_NOMINAL = _obs(goal=(4.0, 0.0))

# Scenario 2: a single near pedestrian triggers the near-human speed cap.
_OBS_SPEED_CAP = _obs(
    goal=(4.0, 0.0),
    ped_positions=[(0.8, 0.3)],
    ped_velocities=[(0.0, 0.0)],
)

# Scenario 3: a static-obstacle wall directly ahead of the robot (rows 9-11,
# columns 11-20) drives the static-clearance / static-collision rejection path
# so that every candidate is rejected and the planner fails closed.
_STATIC_WALL_CELLS = (
    [(10, col) for col in range(11, 21)]
    + [(9, col) for col in range(11, 16)]
    + [(11, col) for col in range(11, 16)]
)
_OBS_STATIC_REJECTION = _obs_with_grid(goal=(4.0, 0.0), occupied_cells=_STATIC_WALL_CELLS)

# Scenario 4: an opt-in goal-posterior channel activates the yield-aside path.
_GOAL_POSTERIOR_CONFIG = HybridRuleLocalPlannerConfig(
    goal_posterior_avoidance_enabled=True,
    goal_posterior_min_confidence=0.5,
    goal_posterior_near_distance=2.0,
    goal_posterior_crossing_lateral_margin=0.5,
    goal_posterior_yield_speed=0.05,
    goal_posterior_turn_rate=0.7,
    goal_posterior_score_bonus=20.0,
)
_OBS_GOAL_POSTERIOR = _obs(
    goal=(4.0, 0.0),
    ped_positions=[(1.0, 0.2)],
    ped_velocities=[(0.0, 0.8)],
)
_OBS_GOAL_POSTERIOR["info"] = {
    "planner_goal_posterior_channel": {
        "enabled": True,
        "pedestrian_goal_posteriors": {
            "crossing_ped_0": {
                "pedestrian_id": "crossing_ped_0",
                "top_goal_id": "crossing_ped_0_route_goal",
                "top_goal_confidence": 0.9,
                "blocker": None,
            },
        },
    }
}


def _nominal_planner() -> HybridRuleLocalPlannerAdapter:
    return HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())


# ===========================================================================
# 1. Config builder characterization (incl. nested proxemic-costmap expansion)
# ===========================================================================

# Fixed raw-config dict exercising scalar coercion plus the nested
# ``proxemic_costmap`` mapping expansion path.
_RAW_CONFIG_WITH_PROXEMIC = {
    "planner_variant": "hybrid_rule_v0_minimal",
    "max_linear_speed": 1.5,
    "max_angular_speed": 1.0,
    "goal_progress_weight": 5.0,
    "proxemic_costmap": {
        "enabled": True,
        "personal_radius": 0.5,
        "social_radius": 1.5,
        "personal_weight": 2.0,
        "social_weight": 0.5,
        "velocity_elongation_factor": 0.2,
        "max_cost": 8.0,
        "decay_function": "gaussian",
    },
}


def test_config_builder_expands_nested_proxemic_costmap_mapping() -> None:
    """The nested proxemic-costmap mapping must flatten into planner config fields."""
    config = build_hybrid_rule_local_planner_config(copy.deepcopy(_RAW_CONFIG_WITH_PROXEMIC))

    # Scalar overrides pass through type-coerced.
    assert config.planner_variant == "hybrid_rule_v0_minimal"
    assert config.max_linear_speed == pytest.approx(1.5)
    assert config.max_angular_speed == pytest.approx(1.0)
    assert config.goal_progress_weight == pytest.approx(5.0)

    # Nested proxemic-costmap fields are expanded onto the planner config.
    assert config.proxemic_costmap_enabled is True
    assert config.proxemic_costmap_personal_radius == pytest.approx(0.5)
    assert config.proxemic_costmap_social_radius == pytest.approx(1.5)
    assert config.proxemic_costmap_personal_weight == pytest.approx(2.0)
    assert config.proxemic_costmap_social_weight == pytest.approx(0.5)
    assert config.proxemic_costmap_velocity_elongation_factor == pytest.approx(0.2)
    assert config.proxemic_costmap_max_cost == pytest.approx(8.0)
    assert config.proxemic_costmap_decay_function == "gaussian"


def test_config_builder_does_not_mutate_input_raw_dict() -> None:
    """The builder must deep-copy its input so callers keep the nested mapping."""
    raw = copy.deepcopy(_RAW_CONFIG_WITH_PROXEMIC)
    build_hybrid_rule_local_planner_config(raw)

    assert "proxemic_costmap" in raw, "builder must not strip the caller's input mapping"
    assert raw["proxemic_costmap"]["enabled"] is True


def test_config_builder_returns_default_for_non_mapping_input() -> None:
    """A non-dict input must fail closed to the default config."""
    config = build_hybrid_rule_local_planner_config(None)

    assert isinstance(config, HybridRuleLocalPlannerConfig)
    assert config == HybridRuleLocalPlannerConfig()


# ===========================================================================
# 2. plan() golden values for the four fixed scenarios
# ===========================================================================


def test_plan_nominal_route_following() -> None:
    """Open-space route-following selects a bounded forward dynamic-window command."""
    command = _nominal_planner().plan(_OBS_NOMINAL)

    assert command == (1.2, 0.0)


def test_plan_near_pedestrian_speed_cap() -> None:
    """A pedestrian inside the slow band caps the linear speed to ``very_slow_speed``."""
    planner = _nominal_planner()

    command = planner.plan(_OBS_SPEED_CAP)

    assert command == (0.075, 0.0)
    last = planner.last_decision()
    assert last is not None
    assert last["planner_mode"] == "NORMAL"
    assert last["selected_source"] == "dynamic_window"
    # The pedestrian at (0.8, 0.3) is sqrt(0.73) ~= 0.8544 m away, inside the
    # 1.0 m slow band, so the command must obey the very-slow speed cap.
    assert last["nearest_pedestrian_distance"] == pytest.approx(0.8544003745317532)
    assert last["nearest_pedestrian_distance"] < planner.config.slow_distance_human
    assert command[0] <= planner.config.very_slow_speed + 1e-9


def test_plan_static_clearance_rejection_fails_closed() -> None:
    """A blocking static-obstacle wall rejects every candidate to an emergency stop."""
    planner = _nominal_planner()

    command = planner.plan(_OBS_STATIC_REJECTION)

    assert command == (0.0, 0.0)
    last = planner.last_decision()
    assert last is not None
    assert last["planner_mode"] == "EMERGENCY_STOP"
    assert last["selected_source"] == "all_candidates_rejected"
    # Both static-clearance and static-collision rejections must fire.
    assert last["rejection_counts"]["static_clearance"] == 48
    assert last["rejection_counts"]["static_collision"] == 18


def test_plan_goal_posterior_active() -> None:
    """An active goal-posterior channel overrides route-following with a yield command."""
    planner = HybridRuleLocalPlannerAdapter(_GOAL_POSTERIOR_CONFIG)

    command = planner.plan(_OBS_GOAL_POSTERIOR)

    assert command == (0.05, -0.7)
    last = planner.last_decision()
    assert last is not None
    assert last["selected_source"] == "goal_posterior_yield_right"
    assert last["goal_posterior_avoidance"]["consumed"] is True
    assert last["goal_posterior_avoidance"]["active"] is True


# ===========================================================================
# 3. diagnostics() / last_decision() dict-shape characterization
# ===========================================================================

# Canonical key sets pinned from main at the time of the lock. The follow-up
# decomposition must preserve these public observation surfaces.
_LAST_DECISION_KEYS = frozenset(
    {
        "corridor_subgoal",
        "goal_posterior_avoidance",
        "moving_rejection_counts",
        "nearest_pedestrian_distance",
        "nearest_static_obstacle_distance",
        "planner_mode",
        "planner_variant",
        "predicted_ttc",
        "progress_windows",
        "proxemic_costmap",
        "rejected_examples",
        "rejection_counts",
        "rejection_counts_by_source",
        "route_corridor",
        "route_trace_recovery",
        "selected_actuation_diagnostics",
        "selected_command",
        "selected_score",
        "selected_source",
        "selected_static_safety_gate",
        "selected_terms",
        "top_k",
        "unavailable_counts",
        "unavailable_examples",
        "value_scorer",
    }
)
_DIAGNOSTICS_KEYS = frozenset(
    {
        "actuation_scoring",
        "fallback_count",
        "goal_posterior_avoidance",
        "last_decision",
        "planner_variant",
        "proxemic_costmap",
        "rejection_counts",
        "selected_source_counts",
        "steps",
        "unavailable_counts",
        "value_scorer",
    }
)


@pytest.mark.parametrize(
    ("label", "planner", "observation"),
    [
        ("nominal", _nominal_planner(), _OBS_NOMINAL),
        ("speed_cap", _nominal_planner(), _OBS_SPEED_CAP),
        ("static_rejection", _nominal_planner(), _OBS_STATIC_REJECTION),
        (
            "goal_posterior",
            HybridRuleLocalPlannerAdapter(_GOAL_POSTERIOR_CONFIG),
            _OBS_GOAL_POSTERIOR,
        ),
    ],
)
def test_diagnostics_and_last_decision_shape_after_plan(label, planner, observation) -> None:
    """Each scenario must expose the pinned diagnostics / last_decision shape."""
    command = planner.plan(observation)

    last = planner.last_decision()
    diagnostics = planner.diagnostics()

    # last_decision mirrors the returned command exactly.
    assert last is not None
    assert last["selected_command"] == [float(command[0]), float(command[1])]
    assert last["planner_variant"] == planner.config.planner_variant

    # Pinned public key sets.
    assert frozenset(last.keys()) == _LAST_DECISION_KEYS
    assert frozenset(diagnostics.keys()) == _DIAGNOSTICS_KEYS

    # Rejection-count structure is always a dict and is aggregated identically
    # at step and episode granularity after a single plan call.
    assert isinstance(last["rejection_counts"], dict)
    assert isinstance(last["moving_rejection_counts"], dict)
    assert isinstance(last["rejection_counts_by_source"], dict)
    assert last["rejection_counts"] == diagnostics["rejection_counts"]
    assert diagnostics["steps"] == 1
    assert diagnostics["fallback_count"] == (1 if label == "static_rejection" else 0)

    # The unavailable-count structure pins the corridor-subgoal-disabled marker.
    assert isinstance(last["unavailable_counts"], dict)
    assert isinstance(last["unavailable_examples"], list)
    assert last["unavailable_counts"] == diagnostics["unavailable_counts"]
    assert last["unavailable_counts"] == {"corridor_subgoal": 1}
    assert last["unavailable_examples"] == [{"source": "corridor_subgoal", "reason": "disabled"}]


def test_last_decision_returns_independent_copy() -> None:
    """``last_decision()`` must return a defensive copy each call."""
    planner = _nominal_planner()
    planner.plan(_OBS_NOMINAL)

    first = planner.last_decision()
    assert first is not None
    original_mode = first["planner_mode"]
    first["planner_mode"] = "mutated"

    assert planner.last_decision()["planner_mode"] == original_mode


# ===========================================================================
# Determinism lock: the four golden commands must reproduce across instances.
# ===========================================================================


def test_characterization_is_deterministic_across_instances() -> None:
    """Two fresh planner instances must produce identical commands for each scenario."""
    expected = [
        ("nominal", _nominal_planner, _OBS_NOMINAL, (1.2, 0.0)),
        ("speed_cap", _nominal_planner, _OBS_SPEED_CAP, (0.075, 0.0)),
        ("static_rejection", _nominal_planner, _OBS_STATIC_REJECTION, (0.0, 0.0)),
        (
            "goal_posterior",
            lambda: HybridRuleLocalPlannerAdapter(_GOAL_POSTERIOR_CONFIG),
            _OBS_GOAL_POSTERIOR,
            (0.05, -0.7),
        ),
    ]

    for label, factory, observation, golden in expected:
        first = factory().plan(observation)
        second = factory().plan(observation)
        assert first == golden, f"{label}: first run {first} != golden {golden}"
        assert second == golden, f"{label}: second run {second} != golden {golden}"
        assert first == second, f"{label}: not deterministic across instances"
