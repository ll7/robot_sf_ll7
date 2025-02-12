import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import timeit

from robot_sf.data_analysis.generate_dataset import extract_key_from_json


def old_function():
    # Find all JSON files in the directory
    json_files = glob.glob("dataset/*.json")

    # Initialize lists to store combined data
    all_acc = []
    all_steering_angles = []
    all_velocities = []
    all_orientations = []

    # Read and combine data from all JSON files
    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)
            acc = [item[0][0] for item in data]
            steering_angles = [item[0][1] for item in data]
            velocities = [item[1][0] for item in data]
            orientations = [item[1][1] for item in data]
            all_acc.append(acc)
            all_steering_angles.append(steering_angles)
            all_velocities.append(velocities)
            all_orientations.append(orientations)

    # Plot acceleration graph
    plt.figure(figsize=(12, 6))

    for i, acceleration in enumerate(all_acc):
        plt.plot(acceleration, label=f"Acceleration {i+1}")

    plt.xlabel("Time Step")
    plt.ylabel("Acceleration")
    plt.title("Acceleration over Time")
    plt.legend()

    # Save the acceleration plot to a file
    plt.tight_layout()
    plt.savefig("dataset/acceleration_plots.png")

    # Plot steering angle graph
    plt.figure(figsize=(12, 6))

    for i, steering_angles in enumerate(all_steering_angles):
        plt.plot(steering_angles, label=f"Steering Angle {i+1}")

    plt.xlabel("Time Step")
    plt.ylabel("Steering Angle")
    plt.title("Steering Angle over Time")
    plt.legend()

    # Save the steering angle plot to a file
    plt.tight_layout()
    plt.savefig("dataset/steering_angle_plots.png")

    # Plot velocity graph
    plt.figure(figsize=(12, 6))

    for i, velocities in enumerate(all_velocities):
        plt.plot(velocities, label=f"Velocity {i+1}")

    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.title("Velocity over Time")
    plt.legend()

    # Save the velocity plot to a file
    plt.tight_layout()
    plt.savefig("dataset/velocity_plots.png")

    # Plot orientation graph
    plt.figure(figsize=(12, 6))

    for i, orientations in enumerate(all_orientations):
        plt.plot(orientations, label=f"Orientation {i+1}")

    plt.xlabel("Time Step")
    plt.ylabel("Orientation")
    plt.title("Orientation over Time")
    plt.legend()

    # Save the orientation plot to a file
    plt.tight_layout()
    plt.savefig("dataset/orientation_plots.png")


def plot_all_npc_ped_positions(filename: str):
    """Plot all NPC pedestrian positions from the given JSON file."""

    # Extract pedestrian positions
    ped_positions = extract_key_from_json(filename, "pedestrian_positions")
    ped_positions_array = np.array(ped_positions)

    x_vals = ped_positions_array[:, :, 0]
    y_vals = ped_positions_array[:, :, 1]

    # colormap for better visibility
    colors = np.random.rand(x_vals.shape[0], x_vals.shape[1])

    plt.scatter(x_vals, y_vals, c=colors, alpha=0.5, s=1)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("All recorded npc pedestrian positions")
    # plt.legend()
    plt.tight_layout()
    plt.savefig("robot_sf/data_analysis/plots/all_npc_pedestrian_positions.png")


def plot_all_npc_ped_velocities(filename: str, raw: bool):
    """Plot all NPC pedestrian velocities from the given JSON file."""

    # Extract pedestrian actions
    ped_actions = extract_key_from_json(filename, "ped_actions")

    velocity_list = []
    for timestep, actions in enumerate(ped_actions):
        if actions:  # Check if positions list is not empty
            current_velocity = []
            for action in actions:
                vel_vector = np.array(action[1]) - np.array(action[0])
                velocity = np.linalg.norm(vel_vector)
                if raw:
                    velocity = velocity / 2  # See pedestrian_env.py -> ped_actions = ...
                current_velocity.append(velocity)
        velocity_list.append(current_velocity)
        # if timestep == 100:
        #     break

    for timestep, vels in enumerate(velocity_list):
        plt.scatter([timestep] * len(vels), vels, alpha=0.5, c="blue", s=1)

    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.title("Pedestrian Velocity over Time, raw" if raw else "Pedestrian Velocity over Time")
    # plt.legend()
    plt.tight_layout()
    title = "all_npc_ped_velocities_raw" if raw else "all_npc_ped_velocities"
    plt.savefig(f"robot_sf/data_analysis/plots/{title}.png")


def plot_ego_ped_acceleration(filename: str):
    """Plot the acceleration of the ego pedestrian."""
    ego_ped_acceleration = [
        item["action"][0] for item in extract_key_from_json(filename, "ego_ped_action")
    ]

    plt.plot(ego_ped_acceleration, label="Acceleration")
    plt.xlabel("Timestep")
    plt.ylabel("Acceleration")
    plt.title("Ego Ped Acceleration over Time")
    plt.legend()
    plt.savefig("robot_sf/data_analysis/plots/ego_ped_acc.png")


def plot_ego_ped_velocity(filename):
    """Plot the velocity of the ego pedestrian."""
    ego_ped_acceleration = [
        item["action"][0] for item in extract_key_from_json(filename, "ego_ped_action")
    ]

    ego_ped_velocity = np.cumsum(ego_ped_acceleration)

    # Clip the velocity to a maximum of 3
    ego_ped_velocity = np.clip(ego_ped_velocity, None, 3)

    plt.plot(ego_ped_velocity, label="Velocity")
    plt.xlabel("Timestep")
    plt.ylabel("Velocity")
    plt.title("Ego Ped Velocity over Time")
    plt.legend()
    plt.savefig("robot_sf/data_analysis/plots/ego_ped_vel.png")
    plt.show()


def print_execution_time(function_call: str):
    """Print the execution time of the given function call."""
    execution_time = timeit.timeit(
        function_call,
        globals=globals(),
        number=1,
    )
    print(f"Execution time: {execution_time} seconds")


def main():
    directory = "robot_sf/data_analysis/datasets"
    # f'plot_all_npc_ped_positions(filename="{directory}/2025-02-06_10-24-12.json")'
    # f'plot_all_npc_ped_velocities(filename="{directory}/2025-02-06_10-24-12.json", raw=True)'
    # f'plot_ego_ped_acceleration(filename="{directory}/2025-01-16_11-47-44.json")'
    # f'plot_ego_ped_velocity(filename="{directory}/2025-01-16_11-47-44.json")'

    print_execution_time(f'plot_ego_ped_velocity(filename="{directory}/2025-01-16_11-47-44.json")')


if __name__ == "__main__":
    main()
