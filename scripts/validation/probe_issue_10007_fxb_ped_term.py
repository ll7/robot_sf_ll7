"""Check the legacy SF pedestrian-term lead without changing any release selector."""

import json
from pathlib import Path

import numpy as np
import yaml

from robot_sf.planner.socnav import SocialForcePlannerAdapter, SocNavPlannerConfig
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings


def main():
    """Print a synthetic standing-pedestrian diagnostic using the native drive."""
    config = yaml.safe_load(Path("configs/algos/social_force_release_v0_0_8.yaml").read_text())
    result = {
        "evidence_status": "diagnostic-only",
        "release_ped_version": config["social_force_ped_version"],
    }
    for version in ("legacy_kernel", "surface_v3"):
        planner = SocialForcePlannerAdapter(
            SocNavPlannerConfig(**{**config, "social_force_ped_version": version})
        )
        ped = {
            "positions": np.array([[1.4, 0.0]]),
            "velocities": np.zeros((1, 2)),
            "count": [1],
            "radius": [0.4],
        }
        force = planner._compute_social_force(
            np.zeros(2), np.array([1.0, 0.0]), ped, 0.0, robot_radius=1.0
        )
        drive = DifferentialDriveRobot(DifferentialDriveSettings(radius=1.0))
        drive.reset_state(((0.0, 0.0), 0.0))
        planner.reset()
        minimum = float("inf")
        contact_step = None
        for step in range(80):
            ped["positions"] = np.array([[1.8, 0.0]])
            observation = {
                "robot": {
                    "position": drive.pos,
                    "heading": [drive.pose[1]],
                    "speed": drive.current_speed,
                    "radius": [1.0],
                },
                "goal": {"current": [10.0, 0.0]},
                "pedestrians": ped,
                "sim": {"timestep": [0.1]},
            }
            command = planner.plan(observation)
            drive.apply_action(tuple((np.asarray(command) - drive.current_speed) / 0.1), 0.1)
            clearance = float(np.linalg.norm(np.array(drive.pos) - [1.8, 0.0]) - 1.4)
            minimum = min(minimum, clearance)
            if clearance <= 0.0 and contact_step is None:
                contact_step = step + 1
        result[version] = {
            "contact_force_at_1mps": force.tolist(),
            "weighted_contact_force": (
                force * config.get("social_force_repulsion_weight", 0.8)
            ).tolist(),
            "standing_ped_at_1_8m_min_clearance": minimum,
            "first_contact_step": contact_step,
            "final_position": list(drive.pos),
            "final_speed": list(drive.current_speed),
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
