"""Pedestrian-removal monotonicity for release arms and the SocialForcePlanner baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from robot_sf.api import _benchmark_observation_from_env, _planner_action_to_env_action
from robot_sf.baselines.interface import Observation
from robot_sf.baselines.social_force import SFPlannerConfig, SocialForcePlanner
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.planner import socnav
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sim.sim_config import SimulationSettings
from tests.metamorphic.planner_arms import RELEASE_ARMS, crossing_scene, run_arm_episode

_SEED = 8244
_DT = 0.1
_MAP_SIZE = 20.0


@dataclass(frozen=True, slots=True)
class _PlannerEpisode:
    """Immutable trace from a bounded episode driven by the native planner."""

    positions: tuple[tuple[float, float, float], ...]
    commands: tuple[tuple[float, float], ...]
    actions: tuple[tuple[float, float], ...]
    pedestrian_positions: tuple[tuple[tuple[float, float], ...], ...]
    success: bool
    collision: bool
    # The loop used ``max_steps`` without the environment terminating. This is not
    # the environment's time-limit ``truncated`` flag.
    step_limit_reached: bool
    pedestrian_count: int
    fallback: bool
    fallback_count: int
    fallback_reason: str | None


def _pedestrian_snapshot(env: object) -> tuple[tuple[float, float], ...]:
    """Copy current simulator pedestrian positions."""
    positions = np.asarray(env.simulator.ped_pos, dtype=float).reshape(-1, 2)  # type: ignore[attr-defined]
    return tuple((float(x), float(y)) for x, y in positions)


def _run_social_force_episode(
    map_def: MapDefinition,
    *,
    seed: int = _SEED,
    max_steps: int = 180,
    planner_seed: int | None = None,
    ped_density: float = 0.0,
    noise_std: float = 0.0,
    pedestrians_yield_to_robot: bool = True,
    interaction_weight: float = 0.05,
) -> _PlannerEpisode:
    """Run a real ``RobotEnv`` episode with ``SocialForcePlanner`` and retain its trace.

    The helper uses the same canonical baseline observation and command projection as the
    public ``robot_sf.run_episode`` facade. It is shared by the mirror, removal,
    and replay relations, which assert that no force-kernel fallback occurred.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    planner_seed = seed if planner_seed is None else planner_seed
    explicit_count = len(map_def.single_pedestrians)
    sim_config = SimulationSettings(
        sim_time_in_secs=max_steps * _DT,
        time_per_step_in_secs=_DT,
        goal_radius=0.2,
        ped_density_by_difficulty=[ped_density],
        difficulty=0,
        max_total_pedestrians=max(12, explicit_count),
    )
    if ped_density == 0.0:
        sim_config.population_size = explicit_count
    env_config = RobotSimulationConfig(
        sim_config=sim_config,
        map_pool=MapDefinitionPool(map_defs={"metamorphic": map_def}),
        map_id="metamorphic",
        robot_config=DifferentialDriveSettings(
            radius=0.3,
            max_linear_speed=1.5,
            max_angular_speed=1.5,
            max_linear_accel=5.0,
            max_angular_accel=5.0,
            max_linear_decel=5.0,
        ),
        peds_have_robot_repulsion=pedestrians_yield_to_robot,
    )
    # The factory seeds construction-time crowd sampling, as benchmark runs do.
    env = make_robot_env(config=env_config, seed=seed)
    planner = SocialForcePlanner(
        SFPlannerConfig(
            dt=_DT,
            action_space="velocity",
            desired_speed=0.8,
            v_max=1.0,
            interaction_weight=interaction_weight,
            noise_std=noise_std,
        ),
        seed=planner_seed,
    )
    positions: list[tuple[float, float, float]] = []
    pedestrian_positions: list[tuple[tuple[float, float], ...]] = []
    commands: list[tuple[float, float]] = []
    actions: list[tuple[float, float]] = []
    last_info: dict[str, object] = {}
    terminated = False
    env_truncated = False

    try:
        _observation, last_info = env.reset(seed=seed)
        planner.reset(seed=planner_seed)
        pedestrian_count = len(env.simulator.ped_pos)
        pedestrian_positions.append(_pedestrian_snapshot(env))
        initial_position = np.asarray(env.simulator.robot_poses[0][0], dtype=float)
        initial_heading = float(env.simulator.robot_poses[0][1])
        positions.append((float(initial_position[0]), float(initial_position[1]), initial_heading))
        previous_position: np.ndarray | None = None

        for _ in range(max_steps):
            planner_observation = _benchmark_observation_from_env(env, previous_position)
            if planner_observation is None:
                raise AssertionError("RobotEnv did not expose the canonical planner observation")
            raw_action = planner.step(planner_observation)
            command = (float(raw_action["vx"]), float(raw_action["vy"]))
            env_action = _planner_action_to_env_action(raw_action, planner, env)
            action_values = np.asarray(env_action, dtype=float).reshape(-1)
            if action_values.size < 2:
                raise AssertionError(f"expected a two-component robot action, got {env_action!r}")

            _observation, _reward, terminated, env_truncated, last_info = env.step(env_action)
            pose = env.simulator.robot_poses[0]
            position = np.asarray(pose[0], dtype=float)
            positions.append((float(position[0]), float(position[1]), float(pose[1])))
            pedestrian_positions.append(_pedestrian_snapshot(env))
            commands.append(command)
            actions.append((float(action_values[0]), float(action_values[1])))
            previous_position = np.asarray(planner_observation.robot["position"], dtype=float)
            if terminated or env_truncated:
                break

        metadata = planner.get_metadata()
        diagnostics = metadata.get("planner_diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        return _PlannerEpisode(
            positions=tuple(positions),
            commands=tuple(commands),
            actions=tuple(actions),
            pedestrian_positions=tuple(pedestrian_positions),
            success=bool(last_info.get("success", False)),
            collision=bool(last_info.get("collision", False)),
            step_limit_reached=bool(not terminated and not env_truncated),
            pedestrian_count=pedestrian_count,
            fallback=bool(diagnostics.get("fallback", False)),
            fallback_count=int(diagnostics.get("fallback_count", 0)),
            fallback_reason=(
                str(diagnostics["fallback_reason"])
                if diagnostics.get("fallback_reason") is not None
                else None
            ),
        )
    finally:
        planner.close()
        env.close()


def _short_navigation_scene(
    *, pedestrian_present: bool, mirrored: bool = False, obstacle_present: bool = True
) -> MapDefinition:
    """Build a short route with a nearby pedestrian, optionally reflecting the whole map."""

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        return (point[0], _MAP_SIZE - point[1]) if mirrored else point

    base_spawn = ((3.0, 8.0), (3.0, 8.0), (3.0, 8.0))
    base_goal = ((4.2, 8.0), (4.2, 8.0), (4.2, 8.0))
    spawn = tuple(transform(point) for point in base_spawn)
    goal = tuple(transform(point) for point in base_goal)
    start, finish = transform((3.0, 8.0)), transform((4.2, 8.0))
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[start, finish],
        spawn_zone=spawn,  # type: ignore[arg-type]
        goal_zone=goal,  # type: ignore[arg-type]
    )
    base_obstacle = [(3.3, 6.7), (3.7, 6.7), (3.7, 7.3), (3.3, 7.3)]
    pedestrians = (
        [
            SinglePedestrianDefinition(
                id="nearby",
                start=transform((4.0, 6.5)),
                goal=transform((6.0, 6.5)),
                speed_m_s=0.8,
            )
        ]
        if pedestrian_present
        else []
    )
    return MapDefinition(
        width=_MAP_SIZE,
        height=_MAP_SIZE,
        obstacles=(
            [Obstacle([transform(point) for point in base_obstacle])] if obstacle_present else []
        ),
        robot_spawn_zones=[spawn],
        ped_spawn_zones=[],
        robot_goal_zones=[goal],
        bounds=[
            (transform((0.0, 0.0)), transform((_MAP_SIZE, 0.0))),
            (transform((_MAP_SIZE, 0.0)), transform((_MAP_SIZE, _MAP_SIZE))),
            (transform((_MAP_SIZE, _MAP_SIZE)), transform((0.0, _MAP_SIZE))),
            (transform((0.0, _MAP_SIZE)), transform((0.0, 0.0))),
        ],
        robot_routes=[route],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=pedestrians,
    )


def test_social_force_planner_accepts_empty_and_nonempty_agent_observations() -> None:
    """The real planner returns finite commands for both pedestrian populations."""
    base = {
        "position": [2.0, 10.0],
        "velocity": [0.0, 0.0],
        "goal": [18.0, 10.0],
        "radius": 0.3,
    }
    populations = (
        [],
        [{"position": [5.0, 10.0], "velocity": [0.0, 0.0], "goal": [5.0, 12.0]}],
    )

    for agents in populations:
        planner = SocialForcePlanner(SFPlannerConfig(dt=_DT, action_space="velocity"), seed=_SEED)
        try:
            command = planner.step(Observation(dt=_DT, robot=base, agents=agents))
            assert set(command) == {"vx", "vy"}
            assert np.isfinite(np.asarray(tuple(command.values()), dtype=float)).all()
            metadata = planner.get_metadata()
            diagnostics = metadata["planner_diagnostics"]
            assert metadata["status"] == "ok"
            assert diagnostics["fallback"] is False
            assert diagnostics["fallback_count"] == 0
        finally:
            planner.close()


def test_social_force_success_is_preserved_when_nearby_pedestrian_is_removed() -> None:
    """A solvable nearby-pedestrian episode and its empty counterpart both succeed."""
    with_pedestrian = _run_social_force_episode(
        _short_navigation_scene(pedestrian_present=True, obstacle_present=False),
        seed=_SEED,
        max_steps=30,
    )
    without_pedestrian = _run_social_force_episode(
        _short_navigation_scene(pedestrian_present=False, obstacle_present=False),
        seed=_SEED,
        max_steps=30,
    )

    assert with_pedestrian.pedestrian_count == 1
    assert without_pedestrian.pedestrian_count == 0
    for episode in (with_pedestrian, without_pedestrian):
        assert episode.success
        assert not episode.collision
        assert not episode.step_limit_reached
        assert episode.fallback is False, episode.fallback_reason
        assert episode.fallback_count == 0
    assert not with_pedestrian.success or without_pedestrian.success
    shared_steps = min(len(with_pedestrian.commands), len(without_pedestrian.commands))
    command_differences = np.linalg.norm(
        np.asarray(with_pedestrian.commands[:shared_steps], dtype=float)
        - np.asarray(without_pedestrian.commands[:shared_steps], dtype=float),
        axis=1,
    )
    assert np.max(command_differences) > 1e-6


@pytest.mark.parametrize("arm", RELEASE_ARMS)
def test_removing_crossing_pedestrian_does_not_reduce_release_arm_success(arm: str) -> None:
    """A successful crossing episode stays successful, and no slower, once the pedestrian goes.

    The pedestrian crosses the route when the robot arrives, so the planner must
    react (its commands differ). The premise ``with_pedestrian.success`` is asserted
    first, so the implication is exercised rather than satisfied vacuously.
    """
    if arm == "orca" and socnav.rvo2 is None:
        pytest.skip("rvo2 is required for the native ORCA release arm")
    with_pedestrian = run_arm_episode(
        arm, crossing_scene(pedestrian=True), seed=_SEED, max_steps=150
    )
    without_pedestrian = run_arm_episode(
        arm, crossing_scene(pedestrian=False), seed=_SEED, max_steps=150
    )

    assert len(with_pedestrian.pedestrian_positions[0]) == 1
    assert len(without_pedestrian.pedestrian_positions[0]) == 0
    shared = min(len(with_pedestrian.commands), len(without_pedestrian.commands))
    assert (
        np.max(
            np.abs(
                np.subtract(with_pedestrian.commands[:shared], without_pedestrian.commands[:shared])
            )
        )
        > 1e-3
    ), "the crossing pedestrian must change the planner's commands"

    assert with_pedestrian.success and not with_pedestrian.collision  # premise
    assert without_pedestrian.success and not without_pedestrian.collision
    assert len(without_pedestrian.commands) <= len(with_pedestrian.commands)
