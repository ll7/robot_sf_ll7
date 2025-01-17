"""
Use the existing abilities to record the visualizable sim state.
Load a saved recording and visualize the gaussian_kde for the pedestrian position.
Afterwards, compare one pedestrian_postion with the gaussian_kde and use Kullback-Leibler divergence to compare the two.
"""

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from robot_sf.render.playback_recording import load_states
from robot_sf.data_analysis.recording_analysis import (
    extract_pedestrian_positions,
    kde_plot_grid_creation,
)
from robot_sf.nav.map_config import MapDefinition


def visualize_kde_of_pedestrians_on_map(
    pedestrian_positions: np.ndarray, map_def: MapDefinition
):
    """Visualize KDE for pedestrian positions on a map"""

    # Get map dimensions
    x_min, x_max, y_min, y_max = map_def.get_map_bounds()
    logger.info(f"Map bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

    # Calculate KDE
    # gaussian_kde expects shape (n_features, n_samples) but our data is (n_samples, n_features)
    # Therefore we transpose the array from shape (n_points, 2) to (2, n_points)
    pedestrian_kde = gaussian_kde(pedestrian_positions.T)

    # Create grid based on map bounds
    grid_xx, grid_yy, grid_points = kde_plot_grid_creation(x_min, x_max, y_min, y_max)

    _, ax = plt.subplots(1, 1, figsize=(6, 5))

    kde_vals = pedestrian_kde(grid_points).reshape(
        grid_xx.shape
    )  # 5. Reshape back to 2D for plotting
    ax.contourf(grid_xx, grid_yy, kde_vals, cmap="viridis")

    ax.scatter(
        pedestrian_positions[:, 0], pedestrian_positions[:, 1], alpha=1, s=1, c="red"
    )

    # Plot map obstacles
    # plot_map_obstacles(ax, map_def)
    map_def.plot_map_obstacles(ax)

    ax.set_title("Pedestrian Positions KDE")
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.show()


def main():
    try:
        # Load pedestrian positions (implement according to your data source)
        states, map_def = load_states("examples/recordings/2024-12-06_15-39-44.pkl")

        pedestrian_positions = extract_pedestrian_positions(states)

        if len(pedestrian_positions) < 2:
            logger.error("Need at least 2 positions for comparison")
            return

        logger.info(f"Pedestrian positions shape: {pedestrian_positions.shape}")

        # Visualize KDE
        visualize_kde_of_pedestrians_on_map(pedestrian_positions, map_def)

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
