"""Real release configs/checkpoints and SVG geometry at the diagnosed dev poses.

The numeric oracles come from obstacle faces, not from the clearance evaluator.
No policy step uses evaluation seeds; these are deterministic local rollouts.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig, OccupancyGrid
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner.guarded_ppo import GuardedPPOAdapter, build_guarded_ppo_config
from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter, build_predictive_mppi_config
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, build_risk_dwa_config

ROOT = Path(__file__).resolve().parents[2]
ARMS = ["risk_dwa", "guarded_ppo", "predictive_mppi"]
POSES = [
    (
        "issue_9762_francis2023_intersection_no_gesture_goal_zone_entry_v2.svg",
        [4.742023, 13.366136],
        0.366136,
    ),
    ("issue_9762_francis2023_blind_corner_goal_zone_entry_v2.svg", [5.469873, 25.391564], 0.391564),
    ("issue_9856_francis2023_exiting_elevator_route_width_v1.svg", [15.179183, 9.071847], 0.471847),
]


@pytest.fixture(params=ARMS)
def planner(request):
    """Keep the actual release configuration, including the registered MPPI model."""
    arm = request.param
    config = yaml.safe_load((ROOT / f"configs/algos/{arm}_release_v0_0_8.yaml").read_text())
    if arm == "risk_dwa":
        adapter = RiskDWAPlannerAdapter(build_risk_dwa_config(config))
    elif arm == "guarded_ppo":
        fallback = RiskDWAPlannerAdapter(build_risk_dwa_config(config["fallback_risk_dwa"]))
        adapter = GuardedPPOAdapter(build_guarded_ppo_config(config), fallback_adapter=fallback)
    else:
        adapter = PredictiveMPPIAdapter(build_predictive_mppi_config(config), allow_fallback=False)
        assert adapter._predictor._ensure_model() is not None
    return arm, adapter


def observation(adapter, map_path, position, heading=0.0, phase=0.0, goal=None):
    """Use the production SVG parser, normalization and rasterizer without a sim step."""
    map_def = convert_map(str(map_path))
    lines, polygons = RobotEnv._normalize_obstacles_for_grid(map_def.obstacles, map_def.bounds)
    # Match the benchmark's normal environment-binding interface. Old adapters
    # lack this binding and therefore retain the diagnosed cell-square behavior.
    env = SimpleNamespace(_get_static_grid_obstacles=lambda: (lines, polygons))
    bind = getattr(adapter, "bind_env", None)
    if callable(bind):
        bind(env)
    grid = OccupancyGrid(
        GridConfig(
            resolution=0.2,
            width=32,
            height=32,
            channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED],
            use_ego_frame=True,
            center_on_robot=True,
        )
    )
    grid.generate(
        lines,
        [],
        ((position[0] + phase, position[1] + phase), heading),
        ego_frame=False,
        obstacle_polygons=polygons,
    )
    goal = goal if goal is not None else [position[0] + 4, position[1]]
    return {
        "occupancy_grid": grid.to_observation(),
        "occupancy_grid_meta": grid.metadata_observation(),
        "robot": {
            "position": position,
            "heading": [heading],
            "speed": [0.0],
            "angular_velocity": [0.0],
            "radius": [1.0],
        },
        "goal": {"current": goal, "next": goal},
        "pedestrians": {
            "count": [0],
            "positions": np.empty((0, 2)),
            "velocities": np.empty((0, 2)),
            "radius": 0.4,
        },
    }


def admissible(arm, adapter, obs, command):
    """Exercise the production rollout, including the actual learned MPPI forecast."""
    pos = np.asarray(obs["robot"]["position"])
    heading = obs["robot"]["heading"][0]
    goal = np.asarray(obs["goal"]["current"])
    if arm == "risk_dwa":
        score = adapter._rollout_score(
            robot_pos=pos,
            heading=heading,
            goal=goal,
            command=command,
            ped_pos=np.empty((0, 2)),
            ped_vel=np.empty((0, 2)),
            observation=obs,
            current_speed=0.0,
        )
        return np.isfinite(score)
    if arm == "guarded_ppo":
        return bool(adapter._evaluate_command(obs, command)["safe"])
    future, mask, steps = adapter._predict_future(obs)
    seq = adapter._constant_sequence(command, steps)
    cost = adapter._sequence_rollout(
        seq,
        robot_pos=pos,
        heading=heading,
        goal=goal,
        future=future,
        mask=mask,
        observation=obs,
        anchor_action=command,
    )
    batch_cost = adapter._batch_sequence_rollout(
        seq[None],
        robot_pos=pos,
        heading=heading,
        goal=goal,
        future=future,
        mask=mask,
        observation=obs,
        anchor_action=command,
    )[0]
    assert batch_cost == pytest.approx(cost)
    return cost < adapter.config.invalid_sequence_cost


def test_tdiag_clearance_is_invariant_to_heading_and_grid_phase(planner):
    """The 0.366/0.392/0.472 m poses are outside the unchanged 0.3 m margin."""
    arm, adapter = planner
    for filename, pos, true_clearance in POSES:
        for heading in [0.0, 0.73, 2.99901]:
            for phase in [0.0, 0.07, 0.19]:
                obs = observation(
                    adapter, ROOT / "maps/successor_svg_maps" / filename, pos, heading, phase
                )
                assert adapter._min_obstacle_clearance(
                    np.asarray(pos), observation=obs
                ) == pytest.approx(true_clearance, abs=1e-7)
                assert admissible(arm, adapter, obs, (0.0, 0.1))


def test_real_corridor_widths_preserve_body_and_margin(planner, tmp_path):
    """Vary two named wall faces of the real corridor SVG, parsed normally.

    1.9/2.0 m cannot fit a positive-clearance 2 m body; 2.6/2.7 m can
    fit the full 0.3 m margin. Check an intruding body in every variant too.
    """
    arm, adapter = planner
    for width in [1.9, 2.0, 2.6, 2.7]:
        tree = ET.parse(ROOT / "maps/svg_maps/atomic_corridor_test.svg")
        for element in tree.iter():
            if element.get("id") == "corridor_top":
                element.set("height", str(9.5 - width / 2))
            if element.get("id") == "corridor_bottom":
                element.set("y", str(10 + width / 2))
                element.set("height", str(9.5 - width / 2))
        path = tmp_path / f"corridor_{width}.svg"
        tree.write(path)
        for heading in [0, 0.73, 1.57]:
            for phase in [0, 0.07, 0.19]:
                obs = observation(adapter, path, [10.0, 10.0], heading, phase)
                clearance = adapter._min_obstacle_clearance(np.array([10.0, 10.0]), observation=obs)
                assert clearance == pytest.approx(width / 2 - 1, abs=1e-7)
                # A bound at-rest turn preserves positive clearance, including
                # the MPPI first-step dead band. Neither arm may admit contact.
                assert admissible(arm, adapter, obs, (0, 0.1)) == (width >= 2.6)
                if arm == "predictive_mppi" and width == 2.7:
                    # Just inside the hard margin, retain monotone recovery.
                    shifted = [10.0, 10.050001]
                    recovery_obs = observation(adapter, path, shifted, heading, phase)
                    assert admissible(arm, adapter, recovery_obs, (0, 0.1))
                intruding = [10.0, 10 + width / 2 - 0.99]
                obs = observation(adapter, path, intruding, heading, phase)
                assert not admissible(arm, adapter, obs, (0, 0.1))


@pytest.mark.parametrize("arm", ["risk_dwa", "predictive_mppi"])
def test_below_margin_recovery_turns_but_rejects_decreasing_clearance(arm):
    """Use the actual MPPI corner stall: rotation is feasible, forward is unsafe."""
    config = yaml.safe_load((ROOT / f"configs/algos/{arm}_release_v0_0_8.yaml").read_text())
    adapter = (
        RiskDWAPlannerAdapter(build_risk_dwa_config(config))
        if arm == "risk_dwa"
        else PredictiveMPPIAdapter(build_predictive_mppi_config(config), allow_fallback=False)
    )
    pos = [21.960488, 22.388717]
    obs = observation(
        adapter,
        ROOT / "maps/successor_svg_maps/issue_9762_classic_group_crossing_goal_zone_entry_v2.svg",
        pos,
        0.090950,
        goal=[20.0, 25.0],
    )
    assert admissible(arm, adapter, obs, (0.0, 0.1)), (
        "positive-clearance rotation must permit recovery"
    )
    assert not admissible(arm, adapter, obs, (0.1, 0.0)), "forward moves closer to the corner"
    command = adapter.plan(obs)
    assert command[0] == pytest.approx(0)
    assert abs(command[1]) > 0
    assert adapter.diagnostics()["recovery_command_count"] == 1
    assert not adapter.diagnostics()["no_admissible_command"]


def test_empty_world_guard_uses_wall_clearance_then_progress_for_recovery():
    """Actual overtaking pose: veto unsafe PPO and choose a nonpenetrating DWA turn."""
    config = yaml.safe_load((ROOT / "configs/algos/guarded_ppo_release_v0_0_8.yaml").read_text())
    fallback = RiskDWAPlannerAdapter(build_risk_dwa_config(config["fallback_risk_dwa"]))
    guard = GuardedPPOAdapter(build_guarded_ppo_config(config), fallback_adapter=fallback)
    obs = observation(
        guard,
        ROOT / "maps/successor_svg_maps/issue_9762_classic_overtaking_goal_zone_entry_v2.svg",
        [19.519706, 9.788156],
        0.872014,
        goal=[19.0, 14.0],
    )
    decision = guard.choose_command_decision(obs, (2.0, 0.0))
    assert decision.filtered_action[0] == pytest.approx(0)
    assert decision.filtered_action[1] != 0, "inf pedestrian ties must not force a permanent stop"
    assert decision.intervened and decision.filtered_action != (2.0, 0.0)
    assert decision.selected_evaluation["min_obs_clear"] > 0
    assert guard.diagnostics()["recovery_command_count"] == 1
