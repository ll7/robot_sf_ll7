import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def load_pedestrian_positions(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    # Extract pedestrian positions and ego_pedestrian positions
    ped_positions = [pos for item in data for pos in item["pedestrian_positions"]]
    ego_ped_positions = [item["ego_ped_pose"][0] for item in data]

    # Convert positions to NumPy arrays
    ped_positions_array = np.array(ped_positions)
    ego_ped_positions_array = np.array(ego_ped_positions)

    return ped_positions_array, ego_ped_positions_array

def perform_kde_on_axis(data, bandwidth=0.1):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(data)

    # Create a grid of points to evaluate the density on
    axis_min, axis_max = data.min() - 1, data.max() + 1
    axis_grid = np.linspace(axis_min, axis_max, 100).reshape(-1, 1)

    # Evaluate the density at the grid points
    log_density = kde.score_samples(axis_grid)

    # Convert log density to actual density values
    density = np.exp(log_density)

    return axis_grid, density

def plot_kde(real_data, ego_ped_data):
    # Perform KDE on real data x positions
    x_positions = real_data[:, 0].reshape(-1, 1)
    x_grid_real, density_real_x = perform_kde_on_axis(x_positions)

    # Perform KDE on ego_pedestrian x positions
    x_positions_ego = ego_ped_data[:, 0].reshape(-1, 1)
    x_grid_ego, density_ego_x = perform_kde_on_axis(x_positions_ego)

    # Perform KDE on real data y positions
    y_positions = real_data[:, 1].reshape(-1, 1)
    y_grid_real, density_real_y = perform_kde_on_axis(y_positions)

    # Perform KDE on ego_pedestrian y positions
    y_positions_ego = ego_ped_data[:, 1].reshape(-1, 1)
    y_grid_ego, density_ego_y = perform_kde_on_axis(y_positions_ego)

    # Plotting the results side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot real data x density
    axes[0].plot(x_grid_real, density_real_x, label='Real Data Density (X Position)')
    axes[0].plot(x_grid_ego, density_ego_x, label='Ego Pedestrian Density (X Position)', linestyle='--')
    axes[0].set_title('Real Data and Ego Pedestrian Density (X Position)')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Plot real data y density
    axes[1].plot(y_grid_real, density_real_y, label='Real Data Density (Y Position)', color='orange')
    axes[1].plot(y_grid_ego, density_ego_y, label='Ego Pedestrian Density (Y Position)', color='red', linestyle='--')
    axes[1].set_title('Real Data and Ego Pedestrian Density (Y Position)')
    axes[1].set_xlabel('Y Position')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('dataset/kde_ego_xy_position_comparison.png')

def main():
    filename = "dataset/2025-01-16_11-47-44.json"
    real_data, ego_ped_data = load_pedestrian_positions(filename)

    # Plot KDE comparison on x and y positions
    plot_kde(real_data, ego_ped_data)

if __name__ == "__main__":
    main()