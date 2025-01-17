"""
Use the existing abilities to record the visualizable sim state.
Load a saved recording and visualize the gaussian_kde for the pedestrian position.
Afterwards, compare one pedestrian_postion with the gaussian_kde and use Kullback-Leibler divergence to compare the two.
"""

from collections import defaultdict
from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_view import VisualizableSimState
from robot_sf.nav.map_config import MapDefinition
import numpy as np
from loguru import logger
from typing import List
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd


def extract_pedestrian_positions(states: List[VisualizableSimState]) -> np.ndarray:
    """Extract pedestrian positions from recorded states."""
    pedestrian_positions = []

    for state in states:
        pedestrian_positions.extend(state.pedestrian_positions)

    # validate that pedestrian_positions has the shape (n, 2)
    if len(pedestrian_positions) == 0:
        logger.error("No pedestrian positions found in states")
        return np.array([])
    if not all(len(pos) == 2 for pos in pedestrian_positions):
        logger.error("Invalid pedestrian positions found in states")
        return np.array([])
    logger.info(f"Extracted {len(pedestrian_positions)} pedestrian positions")

    return np.array(pedestrian_positions)


def compute_kde(positions: np.ndarray) -> gaussian_kde:
    """Compute KDE for positions"""
    return gaussian_kde(positions.T)


def kl_divergence_kde(
    kde1: gaussian_kde, kde2: gaussian_kde, grid_points: np.ndarray
) -> float:
    """Calculate KL divergence between two KDEs"""
    p = kde1(grid_points)
    q = kde2(grid_points)
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(p * np.log(p / q))


def compare_position_with_dataset(
    single_position: np.ndarray, dataset_positions: List[np.ndarray]
) -> float:
    """Compare single position with dataset using KL divergence"""
    # Setup evaluation grid
    grid_x, grid_y = np.linspace(-20, 20, 100), np.linspace(-20, 20, 100)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()])

    # Compute KDEs
    single_kde = compute_kde(single_position)
    dataset_kde = compute_kde(np.concatenate(dataset_positions))

    # Visualize KDEs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    kde_vals_single = single_kde(grid_points).reshape(grid_xx.shape)
    ax1.contourf(grid_xx, grid_yy, kde_vals_single, cmap="viridis")
    ax1.scatter(single_position[:, 0], single_position[:, 1], alpha=0.5)
    ax1.set_title("Single Position KDE")
    ax1.axis("equal")

    kde_vals_dataset = dataset_kde(grid_points).reshape(grid_xx.shape)
    ax2.contourf(grid_xx, grid_yy, kde_vals_dataset, cmap="viridis")
    for pos in dataset_positions:
        ax2.scatter(pos[:, 0], pos[:, 1], alpha=0.5)
    ax2.set_title("Dataset KDE")
    ax2.axis("equal")

    kl_div = kl_divergence_kde(single_kde, dataset_kde, grid_points)
    plt.suptitle(f"KL Divergence: {kl_div:.4f}")
    plt.tight_layout()
    plt.show()

    return kl_div


def main():
    try:
        # Load pedestrian positions (implement according to your data source)
        states, _ = load_states("examples/recordings/2024-11-19_20-39-32.pkl")
        pedestrian_positions = extract_pedestrian_positions(states)

        if len(pedestrian_positions) < 2:
            logger.error("Need at least 2 positions for comparison")
            return

        # Split into dataset and single pedestrian position
        dataset_pedestrian_positions = pedestrian_positions[:-1]
        single_pedestrian_position = pedestrian_positions[-1]

        # Compare pedestrian positions
        kl_div = compare_position_with_dataset(
            single_pedestrian_position, dataset_pedestrian_positions
        )
        logger.info(f"KL Divergence: {kl_div:.4f}")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
