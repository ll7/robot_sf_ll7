import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp, kstest


# Directory containing JSON files
directory = "dataset/"

def load_pedestrian_positions(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    # Extract pedestrian positions
    ped_positions = [pos for item in data for pos in item["pedestrian_positions"]]

    # Convert pedestrian positions to a NumPy array
    ped_positions_array = np.array(ped_positions)

    return ped_positions_array


def perform_kde_solo(data):
    # Fit the KernelDensity model (with Gaussian kernel)
    kde = KernelDensity(kernel='gaussian', bandwidth=1)  # Adjust bandwidth as needed
    kde.fit(data)

    # Create a grid of points to evaluate the density on
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))

    # Stack the grid points into a shape suitable for the model (n_samples, 2)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Evaluate the density at the grid points
    log_density = kde.score_samples(grid_points)

    # Convert log density to actual density values
    density = np.exp(log_density).reshape(x_grid.shape)

    # Plotting the result in 3D
    fig = plt.figure()

    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x_grid, y_grid, density, cmap='viridis')
    # ax.set_zlabel('Density')
    # Labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('3D Gaussian Kernel Density Estimate')
    # plt.contourf(x_grid, y_grid, density, cmap='viridis')
    plt.imshow(density, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Density')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Kernel Density Estimation')


    plt.savefig('dataset/density_plot2.png')

def perform_kde(data, bandwidth=1):
    # Fit the KernelDensity model (with Gaussian kernel)
    kde = KernelDensity(kernel='gaussian', bandwidth=1)
    kde.fit(data)

    # Create a grid of points to evaluate the density on
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))

    # Stack the grid points into a shape suitable for the model (n_samples, 2)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Evaluate the density at the grid points
    log_density = kde.score_samples(grid_points)

    # Convert log density to actual density values
    density = np.exp(log_density).reshape(x_grid.shape)

    kde_synthetic = KernelDensity(kernel='gaussian', bandwidth=1)
    # synthetic_data = kde.sample(data.shape[0])
    synthetic_data = kde.sample(data.shape[0])
    kde_synthetic.fit(synthetic_data)

    # Evaluate the density at the grid points
    log_synthetic = kde_synthetic.score_samples(grid_points)
    density_synthetic = np.exp(log_synthetic).reshape(x_grid.shape)

    perform_test(data, synthetic_data, density, density_synthetic)

    return x_grid, y_grid, density, density_synthetic

def perform_test(real_data, synthetic_data, density_real, density_synthetic):
    # Uniform test
    result = kstest(real_data, 'uniform')
    print(f"Uniform test: {result}")

    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = ks_2samp(real_data, synthetic_data)
    print(f"KS test: {ks_statistic, p_value}")


    # Perform Mean Squared Error
    # print(density_real)
    # print("----------------")
    # print(density_synthetic)
    # mse = mean_squared_error(real_data, synthetic_data)
    # density_mse = mean_squared_error(density_real, density_synthetic)
    # print(f"MSE: {mse}, Density MSE: {density_mse}")


def plot_kde_comparison(real_data):
    # Perform KDE on real data
    x_grid_real, y_grid_real, density_real, density_synthetic = perform_kde(real_data)

    # Plotting the results side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot real data density
    axes[0].imshow(density_real, extent=(x_grid_real.min(), x_grid_real.max(), y_grid_real.min(), y_grid_real.max()), origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_title(f'Real Data Density {density_real.shape}')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')

    # Plot synthetic data density
    axes[1].imshow(density_synthetic, extent=(x_grid_real.min(), x_grid_real.max(), y_grid_real.min(), y_grid_real.max()), origin='lower', cmap='viridis', aspect='auto')
    axes[1].set_title(f'Synthetic Data Density {density_synthetic.shape}')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')

    plt.tight_layout()
    plt.savefig('dataset/kde_comparison2.png')



def main():
    filename = "dataset/2025-01-02_20-19-01.json"
    ped_positions_array = load_pedestrian_positions(filename)
    # perform_kde(ped_positions_array)
    plot_kde_comparison(ped_positions_array)


if __name__ == "__main__":
    main()