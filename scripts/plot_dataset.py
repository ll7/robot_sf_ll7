import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import timeit

# Directory containing JSON files
directory = "dataset/"

def old_function():
    # Find all JSON files in the directory
    json_files = glob.glob(f"{directory}/*.json")

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


def plot_ped_positons(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    # Extract pedestrian positions
    ped_positions = [item["pedestrian_positions"] for item in data]

    # Plot pedestrian positions graph
    plt.figure(figsize=(12, 6))

    for timestep, positions in enumerate(ped_positions):
        if positions:  # Check if positions list is not empty
            x = [pos[0] for pos in positions]
            y = [pos[1] for pos in positions]
            plt.scatter(x, y, alpha=0.5, s=1)
        # if timestep == 100:
        #     break

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Pedestrian Positions over Time")
    # plt.legend()
    plt.tight_layout()
    plt.savefig("dataset/pedestrian_positions.png")


def plot_ped_velocities(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    # Extract pedestrian actions
    ped_actions = [item["ped_actions"] for item in data]

    velocity_list = []
    for timestep, actions in enumerate(ped_actions):
        if actions:  # Check if positions list is not empty
            current_velocity = []
            for action in actions:
                vel_vector = np.array(action[1]) - np.array(action[0])
                velocity = np.linalg.norm(vel_vector)
                current_velocity.append(velocity)
        velocity_list.append(current_velocity)
        # if timestep == 100:
        #     break

    for timestep, vels in enumerate(velocity_list):
        plt.scatter([timestep] * len(vels), vels, alpha=0.5 ,c='blue', s=1)

    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.title("Pedestrian Velocity over Time")
    # plt.legend()
    plt.tight_layout()
    plt.savefig("dataset/pedestrian_velocity.png")


def plot_ped_velocities_raw(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    # Extract pedestrian actions
    ped_actions = [item["ped_actions"] for item in data]

    velocity_list = []
    for timestep, actions in enumerate(ped_actions):
        if actions:  # Check if positions list is not empty
            current_velocity = []
            for action in actions:
                vel_vector = np.array(action[1]) / 2
                velocity = np.linalg.norm(vel_vector)
                current_velocity.append(velocity)
        velocity_list.append(current_velocity)
        # if timestep == 100:
        #     break

    for timestep, vels in enumerate(velocity_list):
        plt.scatter([timestep] * len(vels), vels, alpha=0.5 ,c='red', s=1)

    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.title("Pedestrian Velocity over Time, without scaling")
    # plt.legend()
    plt.tight_layout()
    plt.savefig("dataset/pedestrian_velocity_no_scaling.png")


def plot_ego_ped_acceleration(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    ego_ped_acceleration = [item["ego_ped_action"]["action"][0] for item in data]

    plt.plot(ego_ped_acceleration, label="Acceleration")
    plt.xlabel("Timestep")
    plt.ylabel("Acceleration")
    plt.title("Ego Ped Acceleration over Time")
    plt.legend()
    plt.savefig("dataset/ego_ped_acceleration.png")


def plot_ego_ped_velocity(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    ego_ped_acceleration = [item["ego_ped_action"]["action"][0] for item in data]

    ego_ped_velocity = np.cumsum(ego_ped_acceleration)

    # Clip the velocity to a maximum of 3
    ego_ped_velocity = np.clip(ego_ped_velocity, None, 3)

    plt.plot(ego_ped_velocity, label="Velocity")
    plt.xlabel("Timestep")
    plt.ylabel("Velocitx")
    plt.title("Ego Ped Velocity over Time")
    plt.legend()
    plt.savefig("dataset/ego_ped_velocity.png")

def main():
    # plot_ped_positons(filename="dataset/2025-01-02_20-19-01.json")
    # plot_ped_velocities(filename="dataset/2025-01-02_20-19-01.json")
    # execution_time = timeit.timeit('plot_ped_velocities_raw(filename="dataset/2025-01-02_20-19-01.json")', globals=globals(), number=1)
    # print(f"Execution time: {execution_time} seconds")
    plot_ego_ped_acceleration(filename="dataset/2025-01-16_11-47-44.json")
    plot_ego_ped_velocity(filename="dataset/2025-01-16_11-47-44.json")


if __name__ == "__main__":
    main()