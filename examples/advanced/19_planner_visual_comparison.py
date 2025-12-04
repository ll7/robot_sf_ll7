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
from pathlib import Path

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
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.nav.svg_map_parser import load_svg_maps
from robot_sf.planner.socnav import (
    ORCAPlannerAdapter,
    SACADRLPlannerAdapter,
    SamplingPlannerAdapter,
    SocialForcePlannerAdapter,
    SocNavPlannerConfig,
    SocNavPlannerPolicy,
)

MAP_PATH = Path("maps/svg_maps/planner_comparison.svg")
MAP_KEY = MAP_PATH.stem
EXPECTED_OBS_BBOX = [
    (6.0, 7.0, 0.0, 8.0),
    (11.0, 12.0, 4.0, 12.0),
]
EXPECTED_SIZE = (18.0, 12.0)


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


def _obstacle_bboxes(map_def: MapDefinition) -> list[tuple[float, float, float, float]]:
    """Return (xmin, xmax, ymin, ymax) for each obstacle."""

    bboxes: list[tuple[float, float, float, float]] = []
    for obs in map_def.obstacles:
        xs = [p[0] for p in obs.vertices]
        ys = [p[1] for p in obs.vertices]
        bboxes.append((min(xs), max(xs), min(ys), max(ys)))
    return bboxes


def load_comparison_map() -> MapDefinition:
    """Load the SVG-based comparison map and sanity-check basic dimensions."""

    maps = load_svg_maps(str(MAP_PATH), strict=True)
    map_def = maps.get(MAP_KEY)
    if map_def is None:
        raise FileNotFoundError(f"Expected map '{MAP_KEY}' in {MAP_PATH}")

    if (map_def.width, map_def.height) != EXPECTED_SIZE:
        logger.warning(
            "Loaded map size %s differs from expected %s",
            (map_def.width, map_def.height),
            EXPECTED_SIZE,
        )

    bboxes = _obstacle_bboxes(map_def)
    if sorted(bboxes) != sorted(EXPECTED_OBS_BBOX):
        logger.warning(
            "Loaded obstacle bounds %s differ from expected %s",
            bboxes,
            EXPECTED_OBS_BBOX,
        )

    return map_def


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
        map_pool=MapDefinitionPool(map_defs={MAP_KEY: base_map}),
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
    nav = env.simulator.robot_navs[0]
    final_goal = np.asarray(nav.waypoints[-1], dtype=float).copy()
    # Force navigation to target the final goal directly to avoid early success at mid-waypoints
    nav.new_route([tuple(final_goal), tuple(final_goal)])
    # Refresh observation so goal fields reflect the updated route
    obs = env.state.sensors.next_obs()
    robot_path: list[np.ndarray] = [np.asarray(obs["robot"]["position"], dtype=float).copy()]
    ped_paths: list[np.ndarray] = []
    success = False
    route_complete = False

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

            route_complete = bool(info.get("meta", {}).get("is_route_complete", False))
            if terminated or truncated or route_complete:
                success = route_complete
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
        ax.annotate(
            "goal",
            (result.final_goal[0], result.final_goal[1]),
            textcoords="offset points",
            xytext=(6, 6),
            color=color,
            fontsize=9,
            weight="bold",
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
    base_map = load_comparison_map()
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
