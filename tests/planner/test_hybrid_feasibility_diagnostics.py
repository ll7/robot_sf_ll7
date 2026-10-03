"""Real evaluator witnesses for opt-in feasibility diagnostics (#10092)."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)
from tests.planner.test_hybrid_rule_local_planner import _obs


def _planner(**extra):
    cfg = build_hybrid_rule_local_planner_config(
        dict(
            planner_variant="hybrid_rule_v4_clearance_braking",
            debug_candidate_evaluator=True,
            hard_safety_margin=0.2,
            continuous_static_clearance_enabled=True,
            route_guide_enabled=False,
            **extra,
        )
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    planner.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(
                map_def=SimpleNamespace(width=30, height=30),
                get_obstacle_lines=lambda: np.array([[0, 10, 30, 10]]),
            )
        )
    )
    return planner


def test_debug_forced_stop_reports_the_actual_exclusion_radius():
    """A physical 0.10 m clearance is rejected by a 0.20 m comfort margin."""
    planner = _planner()
    obs = _obs(robot=(4, 10.35), goal=(20, 10.35))  # body radius 0.25 m
    planner.plan(obs)
    debug = (planner.last_decision() or {}).get("candidate_evaluator_debug")
    assert debug is not None, "opt-in evaluator evidence missing"
    assert debug["chosen_stop"] and debug["stop_kind"] == "FORCED"
    assert debug["feasible_moving_count"] == 0
    assert debug["best_feasible_moving"] is None
    assert debug["constraints"] == [
        {
            "constraint": "exact_body_wall_overlap",
            "threshold_name": "exclusion_radius_m",
            "threshold": 0.45,
            "rejected": debug["candidate_count"],
        }
    ]


def test_debug_preferred_stop_retains_a_feasible_moving_witness():
    """An intentionally stop-favouring objective must not be called infeasible."""
    planner = _planner(
        goal_progress_weight=0,
        speed_preference_weight=0,
        freezing_weight=0,
        velocity_smoothness_weight=20,
        control_effort_weight=20,
    )
    planner.plan(_obs(robot=(4, 15), goal=(20, 15)))
    debug = (planner.last_decision() or {}).get("candidate_evaluator_debug")
    assert debug is not None, "opt-in evaluator evidence missing"
    assert debug["stop_kind"] == "PREFERRED"
    assert debug["feasible_moving_count"] > 0
    moving = debug["best_feasible_moving"]
    assert moving["command"][0] > 0
    assert moving["score"] < planner.last_decision()["selected_score"]


def test_opt_in_physical_exclusion_passes_clear_doorway_and_rejects_contact():
    """A 1 m body fits a 2.20 m doorway, while wall overlap remains forbidden."""
    from robot_sf.planner.hybrid_rule_local_planner import HybridRuleCandidate

    accepted = []
    for enabled in (False, True):
        planner = _planner(physical_static_exclusion_enabled=enabled)
        planner.bind_env(
            SimpleNamespace(
                simulator=SimpleNamespace(
                    map_def=SimpleNamespace(width=30, height=30),
                    get_obstacle_lines=lambda: np.array([[0, 10, 30, 10], [0, 12.2, 30, 12.2]]),
                )
            )
        )
        obs = _obs(robot=(4, 11.1), goal=(20, 11.1))
        obs["robot"]["radius"] = np.array([1.0])
        state = planner._extract_state(obs)
        result = planner._evaluate_candidate(
            candidate=HybridRuleCandidate(0.3, 0, "forward"),
            observation=obs,
            state=state,
            speed_cap=2,
            nearest_ped=float("inf"),
        )
        accepted.append(result["accepted"])
        if enabled:
            assert result["accepted"], "comfort margin still excludes a physically clear doorway"
            # Move the body 0.15 m toward the wall: radius 1.0 now overlaps by 0.05 m.
            obs["robot"]["position"] = np.array([4, 10.95])
            result = planner._evaluate_candidate(
                candidate=HybridRuleCandidate(0.3, 0, "forward"),
                observation=obs,
                state=planner._extract_state(obs),
                speed_cap=2,
                nearest_ped=float("inf"),
            )
            assert not result["accepted"]
            assert result["continuous_static_collision"]
            assert result["hard_static_clearance"] == 1.0
    assert accepted == [False, True], "comfort margin still excludes a physically clear doorway"


def test_opt_in_clearance_preference_uses_surface_gap():
    """A 0.40 m free surface gap earns 0.40/0.70 clearance preference, not 1."""
    import pytest

    from robot_sf.planner.hybrid_rule_local_planner import HybridRuleCandidate

    planner = _planner(physical_static_exclusion_enabled=True, desired_static_clearance=0.7)
    planner.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(
                map_def=SimpleNamespace(width=6.2, height=6.2),
                get_obstacle_lines=lambda: np.array([[0, 0.6, 6.2, 0.6]]),
            )
        )
    )
    obs = _obs(robot=(2, 2), goal=(5, 2))
    obs["robot"]["radius"] = np.array([1.0])
    grid = np.zeros((3, 31, 31))
    grid[0, 3, :] = 1  # y = 3 * 0.2 = 0.6; centre-to-wall = 1.4 m
    obs["occupancy_grid"] = grid
    obs["occupancy_grid_meta"] = {
        "origin": [0, 0],
        "resolution": [0.2],
        "size": [6.2, 6.2],
        "use_ego_frame": [0],
        "center_on_robot": [0],
        "channel_indices": [0, 1, 2],
        "robot_pose": [2, 2, 0],
    }
    result = planner._evaluate_candidate(
        candidate=HybridRuleCandidate(0.3, 0, "forward"),
        observation=obs,
        state=planner._extract_state(obs),
        speed_cap=2,
        nearest_ped=float("inf"),
    )
    assert result["accepted"]
    assert result["terms"]["static_clearance"] == pytest.approx(4 / 7)


def test_debug_records_dynamic_and_angular_constraints_in_physical_units():
    """Independent pedestrian gates remain hard, even with physical static exclusion."""
    planner = _planner(physical_static_exclusion_enabled=True, control_period=1.2)
    planner.plan(
        _obs(robot=(4, 15), goal=(20, 15), ped_positions=[(4.65, 15)], ped_velocities=[(0, 0)])
    )
    debug = (planner.last_decision() or {}).get("candidate_evaluator_debug")
    assert debug is not None, "opt-in evaluator evidence missing"
    constraints = {r["constraint"]: r for r in debug["constraints"]}
    assert constraints["dynamic_collision"]["threshold"] == 0.7  # 0.25 + 0.25 + 0.20
    assert constraints["excessive_angular_near_human"]["threshold"] == 0.7
    assert debug["constraint_context"]["near_human_activation_distance_m"] == 0.8
    assert debug["stop_kind"] == "FORCED"


def test_debug_records_braking_rejection_with_short_contact_horizon():
    """A contact after the short hard horizon still rejects an unsafe braking tail."""
    planner = _planner(hard_collision_horizon=0.2)
    planner.plan(
        _obs(
            robot=(4, 15),
            goal=(20, 15),
            speed=1.9,
            ped_positions=[(8, 15)],
            ped_velocities=[(-3, 0)],
        )
    )
    debug = (planner.last_decision() or {}).get("candidate_evaluator_debug")
    assert debug is not None, "opt-in evaluator evidence missing"
    constraints = {r["constraint"]: r for r in debug["constraints"]}
    assert constraints["braking_infeasible"]["threshold"] == 0.7
    assert constraints["braking_infeasible"]["rejected"] > 0


@pytest.mark.parametrize("wall_x", [4.05, 4.10])
def test_physical_exclusion_checks_committed_plant_step_and_whole_sweep(wall_x):
    """A thin wall inside the first 0.10 s must reject a clear 0.20 s rollout."""
    from robot_sf.planner.hybrid_rule_local_planner import HybridRuleCandidate
    from robot_sf.robot.differential_drive import (
        DifferentialDriveMotion,
        DifferentialDriveSettings,
        DifferentialDriveState,
    )

    planner = _planner(physical_static_exclusion_enabled=True)
    planner.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(
                map_def=SimpleNamespace(width=30, height=30),
                get_obstacle_lines=lambda: np.array([[wall_x, 9.99, wall_x, 10.01]]),
            )
        )
    )
    obs = _obs(robot=(4, 10), goal=(20, 10), speed=1)
    obs["robot"]["radius"] = np.array([0.01])
    settings = DifferentialDriveSettings()
    wheel_rate = 1 / settings.wheel_radius
    plant = DifferentialDriveState(
        pose=((4, 10), 0), velocity=(1, 0), wheel_speeds=(wheel_rate, wheel_rate)
    )
    DifferentialDriveMotion(settings).move(plant, (0, 0), 0.1)
    assert plant.pose[0] == pytest.approx((4.1, 10))
    assert planner._continuous_static_collision(np.array([4.2, 10]), 0.01) is False
    # In the 4.05 case even the committed endpoint is clear: only the sweep catches it.
    assert planner._continuous_static_collision(np.array(plant.pose[0]), 0.01) == (wall_x == 4.1)
    evaluation = planner._evaluate_candidate(
        candidate=HybridRuleCandidate(1, 0, "forward"),
        observation=obs,
        state=planner._extract_state(obs),
        speed_cap=2,
        nearest_ped=float("inf"),
    )
    assert not evaluation["accepted"], "unchecked first plant interval crosses a physical wall"
    assert evaluation["reason"] == "static_collision"
    assert evaluation["swept_plant_interval"]
    assert evaluation["time"] == pytest.approx(0.1)
    assert evaluation["hard_static_clearance"] == 0.01


def test_goal_validity_flag_keeps_terminal_goal_when_successor_is_absent():
    """The native sensor's absent successor must not target the world origin."""
    planner = _planner(goal_next_validity_enabled=True)
    obs = _obs(robot=(25.6, 3), goal=(26.4, 3.1))
    obs["goal"]["next"] = np.zeros(2)  # SocNavObservation: None -> zero world position
    obs["goal"]["next_valid"] = np.array([0])
    assert planner._extract_state(obs)["goal"] == pytest.approx((26.4, 3.1)), (
        "terminal waypoint replaced by the absent [0, 0] successor"
    )
    # An ordinary successor remains selectable; the fix does not suppress route lookahead.
    obs["goal"]["next"] = np.array([28, 3])
    obs["goal"]["next_valid"] = np.array([1])
    assert planner._extract_state(obs)["goal"] == pytest.approx((28, 3))


def test_successor_validity_is_independent_and_preserves_legitimate_origin():
    from robot_sf.planner.grid_route import GridRoutePlannerAdapter, GridRoutePlannerConfig

    planner = _planner(goal_next_validity_enabled=True)
    obs = _obs(robot=(25.6, 3), goal=(26.4, 3.1))
    obs["goal"]["next"] = np.zeros(2)
    obs["goal"]["next_valid"] = np.array([0], dtype=np.float32)
    assert planner._extract_state(obs)["goal"] == pytest.approx((26.4, 3.1)), (
        "absent successor selected"
    )
    assert not planner.config.physical_static_exclusion_enabled
    obs["goal"]["next_valid"][0] = 1
    assert planner._extract_state(obs)["goal"] == pytest.approx((0, 0)), "valid origin discarded"
    cfg = GridRoutePlannerConfig()
    cfg.goal_next_validity_enabled = True
    route = GridRoutePlannerAdapter(cfg)
    obs["robot"]["position"] = np.array([26.3, 3.1])
    obs["goal"]["next_valid"][0] = 0
    assert route._extract_state(obs)[2] == pytest.approx((26.4, 3.1)), (
        "route guide targets absent successor"
    )
    obs["goal"]["next_valid"][0] = 1
    assert route._extract_state(obs)[2] == pytest.approx((0, 0))
    hybrid_cfg = build_hybrid_rule_local_planner_config(
        {
            "planner_variant": "hybrid_rule_v4_clearance_braking",
            "route_guide_enabled": True,
            "goal_next_validity_enabled": True,
        }
    )
    guide = HybridRuleLocalPlannerAdapter(hybrid_cfg)._route_guide
    assert guide.config.goal_next_validity_enabled, "hybrid route guide loses validity opt-in"


def test_sensor_emits_explicit_validity_only_when_opted_in():
    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.sensor.socnav_observation import SocNavObservationFusion, socnav_observation_space
    from tests.test_socnav_observation import _build_map_def, _build_socnav_simulator

    cfg = RobotSimulationConfig()
    sim = _build_socnav_simulator([])
    default = SocNavObservationFusion(sim, cfg, 4).next_obs()
    assert "next_valid" not in default["goal"]
    assert (
        "next_valid" not in socnav_observation_space(_build_map_def(10, 10), cfg, 4)["goal"].spaces
    )
    cfg.include_goal_next_valid = True
    fusion = SocNavObservationFusion(sim, cfg, 4)
    absent = fusion.next_obs()
    assert "next_valid" in absent["goal"], "opt-in validity bit missing at sensor"
    assert absent["goal"]["next_valid"].tolist() == [0]
    sim.next_goal_pos[0] = np.zeros(2)
    valid = fusion.next_obs()
    assert valid["goal"]["next_valid"].tolist() == [1]
    assert valid["goal"]["next"].tolist() == [0, 0]
    assert socnav_observation_space(_build_map_def(10, 10), cfg, 4)["goal"]["next_valid"].contains(
        valid["goal"]["next_valid"]
    )


def test_debug_unknown_rejection_reason_is_observable_without_crashing():
    planner = _planner()
    assert planner._debug_rejection_constraint({"reason": "future_physical_rule"}) == (
        "unknown:future_physical_rule",
        None,
        None,
    )


def test_physical_sweep_conservatively_covers_grazing_turn_arc():
    from robot_sf.planner.hybrid_rule_local_planner import HybridRuleCandidate

    planner = _planner(physical_static_exclusion_enabled=True, rollout_horizon=0.1)
    # Independent circular arc: v1, omega0.8, dt0.1 => R1.25, half-angle0.04.
    middle = np.array([4 + 1.25 * np.sin(0.04), 15 + 1.25 * (1 - np.cos(0.04))])
    wall_y = middle[1] - 0.0099  # 0.10mm inside the 0.01m swept disc.
    planner.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(
                map_def=SimpleNamespace(width=30, height=30),
                get_obstacle_lines=lambda: np.array(
                    [[middle[0] - 0.00001, wall_y, middle[0] + 0.00001, wall_y]]
                ),
            )
        )
    )
    obs = _obs(robot=(4, 15), goal=(20, 15), speed=1)
    obs["robot"]["radius"] = np.array([0.01])
    obs["robot"]["angular_velocity"] = np.array([0.8])
    r = planner._evaluate_candidate(
        candidate=HybridRuleCandidate(1, 0.8, "grazing"),
        observation=obs,
        state=planner._extract_state(obs),
        speed_cap=2,
        nearest_ped=float("inf"),
    )
    assert not r["accepted"], "midpoint chord misses grazing arc contact"
    assert r["reason"] == "static_collision"
    assert r["arc_padding_m"] >= 1.25 * (1 - np.cos(0.04))


def test_grid_route_successor_validity_survives_config_builder():
    from robot_sf.planner.grid_route import GridRoutePlannerAdapter, build_grid_route_config

    cfg = build_grid_route_config({"goal_next_validity_enabled": True})
    obs = _obs(robot=(26.3, 3.1), goal=(26.4, 3.1))
    obs["goal"]["next"] = np.zeros(2)
    obs["goal"]["next_valid"] = np.array([0], dtype=np.float32)
    assert GridRoutePlannerAdapter(cfg)._extract_state(obs)[2] == pytest.approx((26.4, 3.1)), (
        "route guide selects absent successor"
    )
    obs["goal"]["next_valid"][0] = 1
    assert GridRoutePlannerAdapter(cfg)._extract_state(obs)[2] == pytest.approx((0, 0))


def test_physical_flag_keeps_explicitly_valid_origin_waypoint():
    planner = _planner(physical_static_exclusion_enabled=True, goal_next_validity_enabled=True)
    obs = _obs(robot=(25.6, 3), goal=(26.4, 3.1))
    obs["goal"]["next"] = np.zeros(2)
    obs["goal"]["next_valid"] = np.array([1], dtype=np.float32)
    assert planner._extract_state(obs)["goal"] == pytest.approx((0, 0)), (
        "world-origin guard discards legitimate successor"
    )


@pytest.mark.parametrize("consumer", ["hybrid", "grid"])
def test_goal_validity_survives_real_observation_flattening(consumer):
    from robot_sf.gym_env.robot_env import _flatten_nested_dict_obs
    from robot_sf.gym_env.unified_config import RobotSimulationConfig
    from robot_sf.planner.grid_route import GridRoutePlannerAdapter, build_grid_route_config
    from robot_sf.sensor.socnav_observation import SocNavObservationFusion
    from tests.test_socnav_observation import _build_socnav_simulator

    cfg = RobotSimulationConfig()
    cfg.include_goal_next_valid = True
    sim = _build_socnav_simulator([])
    sim.robots[0].pose = ((1.0, 1.0), 0.0)
    nested = SocNavObservationFusion(sim, cfg, 4).next_obs()
    # Independent interface fixture also isolates old consumers when the old
    # sensor predates the optional field. Current sensor already emits this bit.
    nested["goal"].setdefault("next_valid", np.array([0], dtype=np.float32))
    flat = _flatten_nested_dict_obs(nested)
    current = nested["goal"]["current"]
    if consumer == "hybrid":
        reader = _planner(goal_next_validity_enabled=True, waypoint_switch_distance=15)
        selected = reader._extract_state(flat)["goal"]
    else:
        reader = GridRoutePlannerAdapter(
            build_grid_route_config({"goal_next_validity_enabled": True, "goal_tolerance": 15})
        )
        selected = reader._extract_state(flat)[2]
    assert selected == pytest.approx(current), "flattened observation loses successor validity"
    flat["goal_next_valid"] = np.array([1], dtype=np.float32)
    if consumer == "hybrid":
        assert reader._extract_state(flat)["goal"] == pytest.approx((0, 0))
    else:
        assert reader._extract_state(flat)[2] == pytest.approx((0, 0))


def test_map_observation_bridge_preserves_optional_successor_validity():
    from robot_sf.benchmark.map_runner.map_runner_observations import normalize_map_observation
    from robot_sf.gym_env.robot_env import _flatten_nested_dict_obs

    default = _flatten_nested_dict_obs(_obs())
    assert "next_valid" not in normalize_map_observation(default)["goal"]
    default["goal_next_valid"] = np.array([0], dtype=np.float32)
    normalized = normalize_map_observation(default)
    assert "next_valid" in normalized["goal"], (
        "map observation bridge drops opt-in successor validity"
    )
    assert normalized["goal"]["next_valid"] is default["goal_next_valid"]
