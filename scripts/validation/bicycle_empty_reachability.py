"""Witness physical reachability of empty-world T60 goals, without planner rankings."""

import json
import math
import os
import pathlib

from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings

rows = []
for steer in [0.52, 0.79]:
    for bearing in [0, 45, 90, 135, 180]:
        r = BicycleDriveRobot(
            BicycleDriveSettings(
                radius=0.64,
                wheelbase=0.9,
                max_steer=steer,
                max_velocity=1.34,
                max_accel=1.0,
                max_decel=1.0,
                allow_backwards=False,
            )
        )
        a = PlannerActionAdapter(r, r.action_space, 0.1)
        goal = (6 * math.cos(math.radians(bearing)), 6 * math.sin(math.radians(bearing)))
        trace = []
        for step in range(600):
            distance = math.dist(r.pos, goal)
            if distance <= 1.64:
                break
            dx, dy = goal[0] - r.pos[0], goal[1] - r.pos[1]
            error = math.atan2(
                math.sin(math.atan2(dy, dx) - r.pose[1]), math.cos(math.atan2(dy, dx) - r.pose[1])
            )
            cmd = (1.34, 2.0 * error)
            action = a.from_velocity_command(cmd)
            r.apply_action(tuple(action), 0.1)
            trace.append(
                {
                    "pose": r.pose,
                    "v": r.state.velocity,
                    "yaw": r.current_yaw_rate,
                    "action": action.tolist(),
                }
            )
        rows.append(
            {
                "steer": steer,
                "bearing": bearing,
                "reached": distance <= 1.64,
                "steps": step,
                "distance": distance,
                "trace": trace,
            }
        )
assert all(r["reached"] for r in rows)
(pathlib.Path(os.environ["BIKEFIX_OUTPUT"]) / "empty-physical-reachability.json").write_text(
    json.dumps(rows, default=float)
)
print([(r["steer"], r["bearing"], r["steps"]) for r in rows])
