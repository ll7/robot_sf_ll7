import matplotlib.pyplot as plt
import numpy as np
import timeit

from robot_sf.data_analysis.generate_dataset import (
    extract_key_from_json,
    extract_key_from_json_as_ndarray,
)


def plot_all_npc_ped_positions(filename: str):
    """Plot all NPC pedestrian positions from the given JSON file."""

    # Extract pedestrian positions
    ped_positions_array = extract_key_from_json_as_ndarray(filename, "pedestrian_positions")

    # E.g.: For only 100 timesteps x_vals = ped_positions_array[:100, :, 0]
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


def plot_all_npc_ped_velocities(filename: str, raw: bool = True):
    """
    Plot all NPC pedestrian velocities from the given JSON file.
    Based on the actions of the npc pedestrians.

    Args: raw (bool): If raw is True, dont use the applied scaling factor (for visual purpose).
    """

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

    ego_ped_velocity = []
    cumulative_sum = 0.0
    for acc in ego_ped_acceleration:
        cumulative_sum += acc
        # Clip, because the ego pedestrian can't go faster than 3 m/s
        cumulative_sum = np.clip(cumulative_sum, None, 3)
        ego_ped_velocity.append(cumulative_sum)

    ego_ped_velocity = np.array(ego_ped_velocity)

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
    print_execution_time(
        f'plot_ego_ped_acceleration(filename="{directory}/2025-01-16_11-47-44.json")'
    )


if __name__ == "__main__":
    main()
