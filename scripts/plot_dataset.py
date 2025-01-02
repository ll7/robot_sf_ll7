import json
import glob
import matplotlib.pyplot as plt

# Directory containing JSON files
directory = "dataset/"

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
