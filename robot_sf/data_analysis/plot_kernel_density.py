import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from robot_sf.data_analysis.generate_dataset import (
    extract_key_from_json_as_ndarray,
    extract_key_from_json,
)


def plot_kde_on_map(filename: str, bandwidth: float = 1.0):
    """Plot the Kernel Density Estimation of pedestrian positions on a map."""
    peds_data = extract_key_from_json_as_ndarray(filename, "pedestrian_positions").reshape(-1, 2)

    # Fit the KernelDensity model (with Gaussian kernel)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)  # Adjust bandwidth as needed
    kde.fit(peds_data)

    # Create a grid of points to evaluate the density on
    x_min, x_max = peds_data[:, 0].min() - 1, peds_data[:, 0].max() + 1
    y_min, y_max = peds_data[:, 1].min() - 1, peds_data[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Stack the grid points into a shape suitable for the model (n_samples, 2)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Evaluate the density at the grid points
    log_density = kde.score_samples(grid_points)

    # Convert log density to actual density values
    density = np.exp(log_density).reshape(x_grid.shape)

    plt.imshow(
        density, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap="viridis", aspect="auto"
    )
    plt.colorbar(label="Density")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Kernel Density Estimation")

    plt.savefig("robot_sf/data_analysis/plots/kde_on_map.png")


def perform_kde_on_axis(data: np.ndarray, bandwidth=0.1):
    """Perform Kernel Density Estimation on a 1D axis."""
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(data)

    # Create a grid of points to evaluate the density on
    axis_min, axis_max = data.min() - 1, data.max() + 1
    axis_grid = np.linspace(axis_min, axis_max, 100).reshape(-1, 1)

    # Evaluate the density at the grid points
    log_density = kde.score_samples(axis_grid)

    # Convert log density to actual density values
    density = np.exp(log_density)

    return axis_grid, density


def plot_kde_in_x_y(filename: str, bandwidth: float = 0.1):
    """Plot the Kernel Density Estimation of npc and ego positions in X and Y axes."""
    peds_data = extract_key_from_json_as_ndarray(filename, "pedestrian_positions").reshape(-1, 2)
    ego_data = np.array([item[0] for item in extract_key_from_json(filename, "ego_ped_pose")])

    # Perform KDE on npc data x positions
    x_positions = peds_data[:, 0].reshape(-1, 1)
    x_grid_npc, density_npc_x = perform_kde_on_axis(x_positions, bandwidth)

    # Perform KDE on ego_pedestrian x positions
    x_positions_ego = ego_data[:, 0].reshape(-1, 1)
    x_grid_ego, density_ego_x = perform_kde_on_axis(x_positions_ego, bandwidth)

    # Perform KDE on npc data y positions
    y_positions = peds_data[:, 1].reshape(-1, 1)
    y_grid_npc, density_npc_y = perform_kde_on_axis(y_positions, bandwidth)

    # Perform KDE on ego_pedestrian y positions
    y_positions_ego = ego_data[:, 1].reshape(-1, 1)
    y_grid_ego, density_ego_y = perform_kde_on_axis(y_positions_ego, bandwidth)

    # Plotting the results side by side
    _, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot npc data x density
    axes[0].plot(x_grid_npc, density_npc_x, label="Npc Data Density (X Position)")
    axes[0].plot(
        x_grid_ego, density_ego_x, label="Ego Pedestrian Density (X Position)", linestyle="--"
    )
    axes[0].set_title("Npc Data and Ego Pedestrian Density (X Position)")
    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Plot npc data y density
    axes[1].plot(y_grid_npc, density_npc_y, label="Npc Data Density (Y Position)", color="orange")
    axes[1].plot(
        y_grid_ego,
        density_ego_y,
        label="Ego Pedestrian Density (Y Position)",
        color="red",
        linestyle="--",
    )
    axes[1].set_title("Npc Data and Ego Pedestrian Density (Y Position)")
    axes[1].set_xlabel("Y Position")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("robot_sf/data_analysis/plots/kde_xy_ego_npc.png")


def main():
    # filename = "robot_sf/data_analysis/datasets/2025-02-06_10-24-12.json"
    filename = "robot_sf/data_analysis/datasets/2025-01-16_11-47-44.json"
    plot_kde_on_map(filename)
    plot_kde_in_x_y(filename)


if __name__ == "__main__":
    main()
