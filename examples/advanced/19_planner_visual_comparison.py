#!/usr/bin/env python3
"""Compare SocNav planner adapters with a side-by-side trajectory overlay.

Usage:
    uv run python examples/advanced/19_planner_visual_comparison.py [--steps 250] [--interactive]

What it does:
    - Runs the same map and seed with four planners (Sampling, Social Force, ORCA, SA-CADRL style)
    - Captures the robot and pedestrian trajectories
    - Produces a matplotlib plot to visually contrast the paths
    - Optional live rendering during rollouts via ``--interactive``

Notes:
    - Matplotlib is required for the overlay; pass ``--interactive`` to watch the runs even
      without plotting.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.planner.socnav import (
    ORCAPlannerAdapter,
    SACADRLPlannerAdapter,
    SamplingPlannerAdapter,
    SocialForcePlannerAdapter,
    SocNavPlannerConfig,
    SocNavPlannerPolicy,
)


@dataclass
class RolloutResult:
    """Container for a planner rollout."""

    name: str
    robot_path: list[np.ndarray]
    ped_paths: list[np.ndarray]
    steps: int
    success: bool
    final_goal: np.ndarray
    start: np.ndarray


def build_comparison_map() -> MapDefinition:
    """Create a compact corridor map with a few interactive pedestrians."""

    width, height = 18.0, 12.0
    obstacles = [
        Obstacle([(6.0, 0.0), (7.0, 0.0), (7.0, 8.0), (6.0, 8.0)]),
        Obstacle([(11.0, 4.0), (12.0, 4.0), (12.0, 12.0), (11.0, 12.0)]),
    ]
    robot_spawn_zones = [((1.0, 1.0), (2.0, 1.0), (2.0, 2.0))]
    robot_goal_zones = [((16.0, 10.0), (17.0, 10.0), (17.0, 11.0))]
    ped_spawn_zones: list[Any] = []
    ped_goal_zones: list[Any] = []
    ped_crowded_zones: list[Any] = []
    bounds = [
        (0.0, width, 0.0, 0.0),
        (0.0, width, height, height),
        (0.0, 0.0, 0.0, height),
        (width, width, 0.0, height),
    ]
    robot_routes = [
        GlobalRoute(
            spawn_id=0,
            goal_id=0,
            # Start the first waypoint far enough from the spawn zone to avoid instant completion
            waypoints=[(4.0, 2.5), (6.5, 6.5), (10.5, 6.5), (15.5, 10.5)],
            spawn_zone=robot_spawn_zones[0],
            goal_zone=robot_goal_zones[0],
        ),
    ]
    ped_routes: list[Any] = []
    single_pedestrians = [
        SinglePedestrianDefinition(id="ped_vertical", start=(9.0, 2.0), goal=(9.0, 11.0)),
        SinglePedestrianDefinition(id="ped_diagonal", start=(3.0, 10.5), goal=(14.5, 3.5)),
        SinglePedestrianDefinition(id="ped_static", start=(7.5, 6.5)),
    ]
    return MapDefinition(
        width,
        height,
        obstacles,
        robot_spawn_zones,
        ped_spawn_zones,
        robot_goal_zones,
        bounds,
        robot_routes,
        ped_goal_zones,
        ped_crowded_zones,
        ped_routes,
        single_pedestrians,
    )


def make_policies(max_speed: float, max_turn: float) -> OrderedDict[str, SocNavPlannerPolicy]:
    """Build planners with shared speed limits for a fair comparison."""

    def cfg() -> SocNavPlannerConfig:
        return SocNavPlannerConfig(
            max_linear_speed=max_speed,
            max_angular_speed=max_turn,
            angular_gain=1.0,
            goal_tolerance=0.25,
        )

    return OrderedDict(
        {
            "sampling": SocNavPlannerPolicy(adapter=SamplingPlannerAdapter(config=cfg())),
            "social_force": SocNavPlannerPolicy(adapter=SocialForcePlannerAdapter(config=cfg())),
            "orca": SocNavPlannerPolicy(adapter=ORCAPlannerAdapter(config=cfg())),
            "sa_cadrl": SocNavPlannerPolicy(adapter=SACADRLPlannerAdapter(config=cfg())),
        },
    )


def rollout_policy(
    name: str,
    policy: SocNavPlannerPolicy,
    base_map: MapDefinition,
    steps: int,
    seed: int,
    interactive: bool,
    max_speed: float,
    max_turn: float,
) -> RolloutResult:
    """Run a single planner on the shared scenario and collect trajectories."""

    env_config = RobotSimulationConfig(
        observation_mode=ObservationMode.SOCNAV_STRUCT,
        map_pool=MapDefinitionPool(map_defs={"planner_comparison": base_map}),
    )
    env_config.sim_config.max_total_pedestrians = len(base_map.single_pedestrians)
    env_config.robot_config.max_linear_speed = max_speed
    env_config.robot_config.max_angular_speed = max_turn

    env = make_robot_env(
        config=env_config,
        seed=seed,
        debug=interactive,
        recording_enabled=False,
        algorithm_name=name,
        scenario_name="planner_comparison",
    )

    obs, _ = env.reset()
    start = np.asarray(obs["robot"]["position"], dtype=float).copy()
    final_goal = np.asarray(env.simulator.robot_navs[0].waypoints[-1], dtype=float).copy()
    robot_path: list[np.ndarray] = [np.asarray(obs["robot"]["position"], dtype=float).copy()]
    ped_paths: list[np.ndarray] = []
    success = False

    try:
        for step_idx in range(steps):
            action = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if interactive:
                env.render()

            robot_path.append(np.asarray(obs["robot"]["position"], dtype=float).copy())
            ped_count = int(obs["pedestrians"]["count"][0])
            ped_paths.append(
                np.asarray(obs["pedestrians"]["positions"][:ped_count], dtype=float).copy(),
            )

            if terminated or truncated or info.get("success", False):
                success = info.get("success", False)
                logger.info(
                    "Planner '{}' finished at step {} (term={} trunc={} success={})",
                    name,
                    step_idx,
                    terminated,
                    truncated,
                    success,
                )
                break
    finally:
        env.exit()

    return RolloutResult(
        name=name,
        robot_path=robot_path,
        ped_paths=ped_paths,
        steps=len(robot_path) - 1,
        success=success,
        final_goal=final_goal,
        start=start,
    )


def plot_rollouts(results: list[RolloutResult], map_def: MapDefinition) -> None:
    """Render overlay plot of robot and pedestrian trajectories."""

    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available; skipping overlay plot.")
        return

    _fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Planner trajectory comparison")
    ax.set_xlim(0, map_def.width)
    ax.set_ylim(0, map_def.height)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Draw map bounds and obstacles
    map_def.plot_map_obstacles(ax)
    ax.set_aspect("equal", adjustable="box")

    colors = ["C0", "C1", "C2", "C3", "C4"]

    for idx, result in enumerate(results):
        path = np.vstack(result.robot_path)
        color = colors[idx % len(colors)]
        ax.plot(path[:, 0], path[:, 1], label=result.name, color=color, linewidth=2)
        ax.scatter(path[0, 0], path[0, 1], color=color, marker="o", s=35, zorder=3)
        ax.scatter(path[-1, 0], path[-1, 1], color=color, marker="x", s=50, zorder=3)
        ax.scatter(
            result.final_goal[0],
            result.final_goal[1],
            color=color,
            marker="*",
            s=80,
            zorder=3,
        )

    # Visualize pedestrian trajectories from the first rollout (shared scenario)
    ped_traces: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for step_peds in results[0].ped_paths:
        for ped_id, pos in enumerate(step_peds):
            ped_traces[ped_id].append((pos[0], pos[1]))
    for trace in ped_traces.values():
        ped_path = np.asarray(trace)
        ax.plot(
            ped_path[:, 0],
            ped_path[:, 1],
            color="0.6",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label="pedestrians" if "pedestrians" not in ax.get_legend_handles_labels()[1] else None,
        )

    ax.legend()
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Visually compare SocNav planner adapters with a shared scenario.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Maximum steps per planner rollout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed applied to each rollout.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Render the simulation live while rolling out each planner.",
    )
    return parser.parse_args()


def main():
    """Entry point for the planner comparison demo."""

    args = parse_args()
    base_map = build_comparison_map()
    max_speed = 1.8
    max_turn = 1.2
    policies = make_policies(max_speed=max_speed, max_turn=max_turn)

    results: list[RolloutResult] = []
    for name, policy in policies.items():
        logger.info("Running planner '{}'", name)
        result = rollout_policy(
            name=name,
            policy=policy,
            base_map=base_map,
            steps=args.steps,
            seed=args.seed,
            interactive=args.interactive,
            max_speed=max_speed,
            max_turn=max_turn,
        )
        results.append(result)
        logger.info(
            "Planner '{name}' steps={steps} success={success}",
            name=name,
            steps=result.steps,
            success=result.success,
        )

    plot_rollouts(results, base_map)


if __name__ == "__main__":
    main()
