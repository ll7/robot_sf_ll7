"""
Use the existing abilities to record the visualizable sim state.
Load a saved recording and visualize the gaussian_kde for the pedestrian trajectories.
Afterwards, compare one trajectory with the gaussian_kde and use Kullback-Leibler divergence to compare the two.
"""

from robot_sf.render.playback_recording import load_states
import numpy as np
import loguru
from typing import List
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd


def extract_pedestrian_trajectory_from_states(states: List[VisualizableSimState]):
    """Extract pedestrian trajectories from a list of VisualizableSimState objects

    Should return a list of trajectories where each trajectory is a tuple of (x_positions, y_positions)
    """
    # TODO: Implement loading from recording file
    # Should return a list of trajectories where each trajectory is a tuple of (x_positions, y_positions)




def compute_kde(x, y):
    """Create a 2D KDE for a trajectory"""
    values = np.vstack([x, y])
    return gaussian_kde(values)


def kl_divergence_kde(kde1, kde2, grid_points):
    """Calculate KL divergence between two KDEs"""
    p = kde1(grid_points)
    q = kde2(grid_points)
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(p * np.log(p / q))


def compare_trajectory_with_dataset(single_trajectory, dataset_trajectories):
    """Compare one trajectory against a dataset of trajectories using KL-divergence"""
    # Setup evaluation grid
    grid_x, grid_y = np.linspace(-4, 4, 100), np.linspace(-4, 4, 100)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()])

    # Compute KDE for single trajectory
    single_kde = compute_kde(single_trajectory[0], single_trajectory[1])

    # Compute KDE for dataset
    dataset_x = np.concatenate([traj[0] for traj in dataset_trajectories])
    dataset_y = np.concatenate([traj[1] for traj in dataset_trajectories])
    dataset_kde = compute_kde(dataset_x, dataset_y)

    # Calculate KL divergence
    kl_div = kl_divergence_kde(single_kde, dataset_kde, grid_points)

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot single trajectory KDE
    kde_vals_single = single_kde(grid_points).reshape(grid_xx.shape)
    ax1.contourf(grid_xx, grid_yy, kde_vals_single, cmap="viridis")
    ax1.set_title("Single Trajectory KDE")

    # Plot dataset KDE
    kde_vals_dataset = dataset_kde(grid_points).reshape(grid_xx.shape)
    ax2.contourf(grid_xx, grid_yy, kde_vals_dataset, cmap="viridis")
    ax2.set_title("Dataset KDE")

    plt.suptitle(f"KL Divergence: {kl_div:.4f}")
    plt.show()

    return kl_div


def main():
    # Load recorded trajectories
    recording_path = "path_to_recording.pkl"  # Update with actual path

    states, map_def = load_states(recording_path)

    trajectories = extract_pedestrian_trajectory_from_states(states)

    # Split into dataset and single trajectory for comparison
    dataset_trajectories = trajectories[:-1]  # All but last trajectory
    single_trajectory = trajectories[-1]  # Last trajectory

    # Compare
    kl_div = compare_trajectory_with_dataset(single_trajectory, dataset_trajectories)
    print(f"KL Divergence between single trajectory and dataset: {kl_div}")


if __name__ == "__main__":
    main()
