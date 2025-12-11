"""Advanced occupancy grid workflow: standalone grids, queries, and reward shaping.

What this demo covers:
1) Builds a standalone ego-frame grid with synthetic obstacles/pedestrians
2) Runs spawn-validation queries (point + circular regions) with per-channel results
3) Shows a simple occupancy-based reward shaping loop inside RobotEnv
4) Visualizes results with both matplotlib heatmaps and Pygame overlay exports

All steps run headless and finish in a few seconds; no extra assets are required.
Generates visualizations saved to output/plots/ for inspection.
"""

from __future__ import annotations

import math
import os

from loguru import logger

from robot_sf.common import ensure_canonical_tree
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import GridConfig, RobotSimulationConfig
from robot_sf.nav.occupancy_grid import (
    GridChannel,
    OccupancyGrid,
    POIQuery,
    POIQueryType,
)


def build_standalone_grid() -> tuple[OccupancyGrid, list[POIQuery]]:
    """Create and query a grid without spinning up the full environment."""
    grid_config = GridConfig(
        width=6.0,
        height=6.0,
        resolution=0.2,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED],
        use_ego_frame=True,
    )
    grid = OccupancyGrid(config=grid_config)

    obstacles = [((1.0, 1.0), (5.0, 1.0)), ((1.0, 4.5), (5.0, 4.5))]
    pedestrians = [((2.5, 3.0), 0.35), ((3.5, 2.0), 0.3)]
    robot_pose = ((3.0, 3.0), math.pi / 4)

    grid.generate(obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose)

    queries = [
        POIQuery(x=3.0, y=3.0, radius=0.5, query_type=POIQueryType.CIRCLE),
        POIQuery(x=1.2, y=1.0, query_type=POIQueryType.POINT),
    ]
    for query in queries:
        result = grid.query(query)
        per_channel = {ch.value: f"{val:.3f}" for ch, val in result.per_channel_results.items()}
        logger.info(
            f"Query {query.query_type.value} @ ({query.x:.2f}, {query.y:.2f}): "
            f"occupied={result.is_occupied} safe_to_spawn={result.safe_to_spawn} per-channel={per_channel}"
        )

    _render_pygame_overlay(grid, robot_pose)
    _visualize_grid_channels(grid, queries, robot_pose, "standalone_grid")
    return grid, queries


def reward_shaping_demo() -> None:
    """Show how to derive a simple clearance penalty from the grid observation."""
    grid_config = GridConfig(
        width=8.0,
        height=8.0,
        resolution=0.25,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED],
    )
    config = RobotSimulationConfig(
        use_occupancy_grid=True,
        include_grid_in_observation=True,
        grid_config=grid_config,
    )
    env = make_robot_env(config=config, debug=False)

    obs, _info = env.reset(seed=7)
    shaped_return = 0.0

    # Track metrics for visualization
    steps = []
    clearance_penalties = []
    shaped_rewards = []

    for step_idx in range(8):
        grid_obs = obs["occupancy_grid"]
        clearance_penalty = float(grid_obs[-1].max())  # combined channel
        shaped_reward = -0.1 * clearance_penalty
        shaped_return += shaped_reward

        steps.append(step_idx)
        clearance_penalties.append(clearance_penalty)
        shaped_rewards.append(shaped_reward)

        logger.info(
            f"Step {step_idx:02d}: clearance_penalty={clearance_penalty:.3f} shaped_reward={shaped_reward:.3f}"
        )

        action = env.action_space.sample()
        obs, _reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=step_idx + 20)

    env.close()
    logger.info(f"Toy shaped return after 8 steps: {shaped_return:.3f}")

    # Visualize reward shaping timeline
    _visualize_reward_timeline(steps, clearance_penalties, shaped_rewards)


def _render_pygame_overlay(grid: OccupancyGrid, robot_pose) -> None:
    """Render and export grid overlay as PNG via Pygame."""
    try:
        import pygame
    except ImportError:
        logger.warning("Skipping overlay rendering; pygame not installed.")
        return

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    scale = 4
    surface = pygame.Surface((grid.shape[2] * scale, grid.shape[1] * scale), pygame.SRCALPHA)
    grid.render_pygame(surface, robot_pose=robot_pose, scale=scale, alpha=160)
    logger.info(f"Rendered headless grid overlay surface at {surface.get_size()}")

    # Export to PNG
    try:
        output_dir = ensure_canonical_tree(categories=("tmp",))
        output_path = output_dir / "grid_overlay_pygame.png"
        pygame.image.save(surface, str(output_path))
        logger.info(f"Saved Pygame overlay to {output_path}")
    except Exception as e:
        logger.warning("Failed to export Pygame overlay: %s", e)
    finally:
        pygame.quit()


def _visualize_grid_channels(
    grid: OccupancyGrid,
    queries: list[POIQuery],
    robot_pose: tuple,
    name: str = "grid",
) -> None:
    """Visualize grid channels and query results using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping channel visualization.")
        return

    output_dir = ensure_canonical_tree(categories=("tmp",))

    # Extract grid data
    channels = [GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED]

    # Create subplots for each channel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Occupancy Grid Channels: {name}", fontsize=14, fontweight="bold")

    for ax, channel in zip(axes, channels, strict=True):
        channel_data = grid.get_channel(channel)

        im = ax.imshow(channel_data.T, cmap="hot", origin="lower")
        ax.set_title(f"{channel.value} Channel")
        ax.set_xlabel("X (grid cells)")
        ax.set_ylabel("Y (grid cells)")
        plt.colorbar(im, ax=ax, label="Occupancy")

        # Overlay query points if present
        for query in queries:
            # Convert query coordinates to grid indices
            grid_x = int((query.x + grid.config.width / 2) / grid.config.resolution)
            grid_y = int((query.y + grid.config.height / 2) / grid.config.resolution)

            if query.query_type == POIQueryType.POINT:
                ax.plot(grid_x, grid_y, "b*", markersize=15, label="Point Query")
            else:
                circle_radius_cells = int(query.radius / grid.config.resolution)
                circle = plt.Circle(
                    (grid_x, grid_y),
                    circle_radius_cells,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=2,
                    label="Circle Query",
                )
                ax.add_patch(circle)

        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    output_path = output_dir / f"grid_channels_{name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved grid channel visualization to {output_path}")
    plt.close()


def _visualize_reward_timeline(
    steps: list[int],
    clearance_penalties: list[float],
    shaped_rewards: list[float],
) -> None:
    """Plot clearance penalty and reward shaping timeline."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping reward timeline visualization.")
        return

    output_dir = ensure_canonical_tree(categories=("tmp",))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Reward Shaping Timeline", fontsize=14, fontweight="bold")

    # Clearance penalty plot
    ax1.plot(steps, clearance_penalties, "o-", linewidth=2, markersize=6, color="red")
    ax1.fill_between(steps, clearance_penalties, alpha=0.3, color="red")
    ax1.set_ylabel("Clearance Penalty (max occupancy)")
    ax1.set_title("Grid Occupancy Clearance Over Time")
    ax1.grid(True, alpha=0.3)

    # Shaped reward plot
    ax2.plot(steps, shaped_rewards, "s-", linewidth=2, markersize=6, color="blue")
    ax2.fill_between(steps, shaped_rewards, alpha=0.3, color="blue")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Shaped Reward (-0.1 × clearance penalty)")
    ax2.set_title("Derived Reward Signal")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "reward_shaping_timeline.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved reward timeline to {output_path}")
    plt.close()


def main() -> None:
    """Run standalone grid build and reward shaping demos with visualizations."""
    logger.info("Starting occupancy grid workflow with visual analysis...")
    build_standalone_grid()
    reward_shaping_demo()
    logger.info("Workflow complete. Generated visualizations in output/:")
    logger.info("  - grid_overlay_pygame.png: Pygame-rendered grid overlay")
    logger.info(
        "  - grid_channels_standalone_grid.png: Matplotlib heatmaps (obstacles, pedestrians, combined)"
    )
    logger.info("  - reward_shaping_timeline.png: Clearance penalty and reward signals over time")


if __name__ == "__main__":
    main()
