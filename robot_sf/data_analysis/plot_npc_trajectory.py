import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def load_pedestrian_positions(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    # Extract pedestrian positions
    ped_positions = [item["pedestrian_positions"] for item in data]

    ped_actions = [item["ped_actions"] for item in data]

    # Convert positions to NumPy arrays
    ped_positions_array = np.array(ped_positions)
    ped_actions_array = np.array(ped_actions)

    return ped_positions_array


def plot_single_splitted_traj(ped_positions_array, ped_idx=0):
    # ped_positions_array shape: (timesteps, num_pedestrians, 2)

    # x_vals = ped_positions_array[:, :, 0]
    # y_vals = ped_positions_array[:, :, 1]
    # plt.plot(x_vals, y_vals)

    x_vals = ped_positions_array[:, ped_idx, 0]
    y_vals = ped_positions_array[:, ped_idx, 1]

    distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
    counter = 0
    start_idx = 0
    for i, dist in enumerate(distances):
        if dist > 2:  # Threshold for abnormal distance
            plt.plot(
                x_vals[start_idx : i + 1],
                y_vals[start_idx : i + 1],
                label=f"Pedestrian {start_idx}",
            )
            plt.scatter(
                x_vals[start_idx], y_vals[start_idx], color="green", marker="o"
            )  # Start point
            plt.scatter(x_vals[i], y_vals[i], color="red", marker="x")  # End point
            counter += 1
            start_idx = i + 1
    # Plot the last segment
    plt.plot(x_vals[start_idx:], y_vals[start_idx:], label=f"Pedestrian {start_idx}")
    plt.scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")  # Start point
    plt.scatter(x_vals[-1], y_vals[-1], color="red", marker="x")  # End point

    # plt.plot(x_vals, y_vals)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Pedestrian Trajectories: {x_vals.shape[0]}")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig("dataset/single_splitted_npc_traj4.png")


def plot_all_splitted_traj(ped_positions_array):
    timesteps, num_pedestrians, _ = ped_positions_array.shape

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
                plt.scatter(
                    x_vals[start_idx], y_vals[start_idx], color="green", marker="o"
                )  # Start point
                plt.scatter(x_vals[i], y_vals[i], color="red", marker="x")  # End point
                start_idx = i + 1
        # Plot the last segment
        plt.plot(x_vals[start_idx:], y_vals[start_idx:], label=f"Pedestrian {ped_idx} Segment")
        plt.scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")  # Start point
        plt.scatter(x_vals[-1], y_vals[-1], color="red", marker="x")  # End point

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Pedestrian Trajectories")
    plt.legend()
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.savefig("dataset/all_splitted_npc_traj.png")


def calculate_velocity(x_vals, y_vals, time_interval=0.1):
    # Calculate the differences between consecutive points
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)

    # Calculate the Euclidean distance (velocity) between consecutive points
    distances = np.sqrt(dx**2 + dy**2)
    velocities = distances / time_interval

    return velocities


def calculate_acceleration(velocities, time_interval=0.1):
    # Calculate the differences between consecutive velocities
    dv = np.diff(velocities)

    # Calculate the acceleration
    accelerations = dv / time_interval

    return accelerations


def subplot_single_splitted_traj_acc(ped_positions_array, ped_idx=0):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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
            axes[0].scatter(
                x_vals[start_idx], y_vals[start_idx], color="green", marker="o"
            )  # Start point
            axes[0].scatter(x_vals[i], y_vals[i], color="red", marker="x")  # End point

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

    velocities = calculate_velocity(x_vals[start_idx : i + 1], y_vals[start_idx : i + 1])
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
    plt.savefig("dataset/single_splitted_npc_traj7.png")


def main():
    # filename = "dataset/2025-01-16_11-47-44.json"
    # filename = "dataset/2025-01-02_20-19-01.json"
    # filename = "dataset/2025-02-06_10-17-18.json"
    filename = "dataset/2025-02-06_10-24-12.json"
    positions = load_pedestrian_positions(filename)
    # plot_trajectories(real_data)

    # plot_all_splitted_traj(real_data)
    # plot_single_splitted_traj(positions, ped_idx=15)
    subplot_single_splitted_traj_acc(positions, ped_idx=3)


if __name__ == "__main__":
    main()
