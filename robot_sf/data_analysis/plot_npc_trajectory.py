import numpy as np
import matplotlib.pyplot as plt

from robot_sf.data_analysis.generate_dataset import extract_key_from_json_as_ndarray


def plot_single_splitted_traj(filename: str, ped_idx: int = 0):
    """
    Plot from JSON file from a single pedestrian id the multiple trajectories.
    Split when the distance between two consecutive points is greater than normal.

    ped_positions_array shape: (timesteps, num_pedestrians, 2)
    """
    ped_positions_array = extract_key_from_json_as_ndarray(filename, "pedestrian_positions")

    x_vals = ped_positions_array[:, ped_idx, 0]
    y_vals = ped_positions_array[:, ped_idx, 1]

    distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
    start_idx = 0
    for i, dist in enumerate(distances):
        if dist > 2:  # Threshold for abnormal distance
            plt.plot(
                x_vals[start_idx : i + 1],
                y_vals[start_idx : i + 1],
                label=f"Pedestrian {start_idx}",
            )
            # Start point
            plt.scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")
            # End point
            plt.scatter(x_vals[i], y_vals[i], color="red", marker="x")
            start_idx = i + 1

    # Plot the last segment
    plt.plot(x_vals[start_idx:], y_vals[start_idx:], label=f"Pedestrian {start_idx}")
    plt.scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")  # Start point
    plt.scatter(x_vals[-1], y_vals[-1], color="red", marker="x")  # End point

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Pedestrian Trajectories: {x_vals.shape[0]}")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(f"robot_sf/data_analysis/plots/single_splitted_npc{ped_idx}_traj.png")


def plot_all_splitted_traj(filename: str):
    """
    Plot from JSON file all npc pedestrian trajectories.
    Split when the distance between two consecutive points is greater than normal.

    ped_positions_array shape: (timesteps, num_pedestrians, 2)
    """
    ped_positions_array = extract_key_from_json_as_ndarray(filename, "pedestrian_positions")
    _, num_pedestrians, _ = ped_positions_array.shape

    for ped_idx in range(num_pedestrians):
        # Extract x and y for this ped_idx across all timesteps
        x_vals = ped_positions_array[:, ped_idx, 0]
        y_vals = ped_positions_array[:, ped_idx, 1]

        distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
        start_idx = 0
        for i, dist in enumerate(distances):
            if dist > 2:  # Threshold for abnormal distance
                plt.plot(
                    x_vals[start_idx : i + 1],
                    y_vals[start_idx : i + 1],
                    label=f"Pedestrian {ped_idx} Segment",
                )
                # Start point
                plt.scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")
                # End point
                plt.scatter(x_vals[i], y_vals[i], color="red", marker="x")
                start_idx = i + 1

        # Plot the last segment
        plt.plot(x_vals[start_idx:], y_vals[start_idx:], label=f"Pedestrian {ped_idx} Segment")
        plt.scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")
        plt.scatter(x_vals[-1], y_vals[-1], color="red", marker="x")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Pedestrian Trajectories")
    # plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig("robot_sf/data_analysis/plots/all_splitted_npc_traj.png")


def calculate_velocity(
    x_vals: np.ndarray, y_vals: np.ndarray, time_interval: float = 0.1
) -> np.ndarray:
    """Calculate the velocity of a pedestrian given their x and y positions."""
    # Calculate the differences between consecutive points
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)

    # Calculate the Euclidean distance (velocity) between consecutive points
    distances = np.sqrt(dx**2 + dy**2)
    velocities = distances / time_interval

    return velocities


def calculate_acceleration(velocities: np.ndarray, time_interval: float = 0.1) -> np.ndarray:
    """Calculate the acceleration of a pedestrian given their velocities."""
    # Calculate the differences between consecutive velocities
    dv = np.diff(velocities)

    # Calculate the acceleration
    accelerations = dv / time_interval

    return accelerations


def subplot_single_splitted_traj_acc(filename: str, ped_idx: float = 0):
    """
    Plot from JSON file for a single pedestrian id trajectories, velocity and acceleration.
    """
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    ped_positions_array = extract_key_from_json_as_ndarray(filename, "pedestrian_positions")

    x_vals = ped_positions_array[:, ped_idx, 0]
    y_vals = ped_positions_array[:, ped_idx, 1]

    distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
    counter = 0
    start_idx = 0
    for i, dist in enumerate(distances):
        if dist > 2:  # Threshold for abnormal distance
            axes[0].plot(
                x_vals[start_idx : i + 1],
                y_vals[start_idx : i + 1],
                label=f"Pedestrian {start_idx}",
            )
            axes[0].scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")
            axes[0].scatter(x_vals[i], y_vals[i], color="red", marker="x")

            velocities = calculate_velocity(x_vals[start_idx : i + 1], y_vals[start_idx : i + 1])
            accelerations = calculate_acceleration(velocities)
            axes[1].plot(range(len(velocities)), velocities, label=f"Pedestrian {start_idx}")
            axes[2].plot(range(len(accelerations)), accelerations, label=f"Pedestrian {start_idx}")

            counter += 1
            start_idx = i + 1
    # Plot the last segment
    axes[0].plot(x_vals[start_idx:], y_vals[start_idx:], label=f"Pedestrian {start_idx}")
    axes[0].scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")  # Start point
    axes[0].scatter(x_vals[-1], y_vals[-1], color="red", marker="x")  # End point

    velocities = calculate_velocity(x_vals[start_idx:], y_vals[start_idx:])
    accelerations = calculate_acceleration(velocities)
    axes[1].plot(range(len(velocities)), velocities, label=f"Pedestrian {start_idx}")
    axes[2].plot(range(len(accelerations)), accelerations, label=f"Pedestrian {start_idx}")

    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Y Position")
    axes[0].set_title(f"Pedestrian Trajectories: {x_vals.shape[0]}")
    axes[0].invert_yaxis()
    axes[0].legend()

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Velocity")
    axes[1].set_title("Pedestrian Velocities")
    axes[1].legend()

    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Acceleration")
    axes[2].set_title("Pedestrian Accelerations")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"robot_sf/data_analysis/plots/subplot_npc_{ped_idx}.png")


def main():
    filename = "robot_sf/data_analysis/datasets/2025-02-06_10-24-12.json"
    plot_all_splitted_traj(filename)
    # plot_single_splitted_traj(filename, ped_idx=15)
    # subplot_single_splitted_traj_acc(filename, ped_idx=3)


if __name__ == "__main__":
    main()
