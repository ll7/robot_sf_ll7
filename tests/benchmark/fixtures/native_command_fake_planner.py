#!/usr/bin/env python3
"""Fake native planner for issue #5887 native-command arm tests.

Reads one JSON request per line from stdin (persistent mode) or from a single
stdin read (per-episode mode) and replies with a goal-directed unicycle command
computed from the request's ``robot`` and ``goal`` fields. It never actually
solves a plan; it only satisfies the native-command request/response contract so
the benchmark runner's subprocess arm can be exercised on CPU without a real
planner binary.

Exit code is 0 on success. Malformed input would be a contract failure, but this
fixture always succeeds for the supported request shape.
"""

from __future__ import annotations

import json
import math
import sys


def _command(robot: dict, goal: dict) -> dict[str, float]:
    robot_pos = robot.get("position") or [0.0, 0.0]
    goal_pos = goal.get("current") or [0.0, 0.0]
    dx = float(goal_pos[0]) - float(robot_pos[0])
    dy = float(goal_pos[1]) - float(robot_pos[1])
    dist = math.hypot(dx, dy)
    heading = (
        float(robot.get("heading", [0.0])[0]) if isinstance(robot.get("heading"), list) else 0.0
    )
    if dist < 1e-6:
        return {"linear_velocity": 0.0, "angular_velocity": 0.0}
    desired = math.atan2(dy, dx)
    err = (desired - heading + math.pi) % (2 * math.pi) - math.pi
    angular = max(-1.0, min(1.0, err))
    linear = max(0.0, min(1.0, dist)) * (1.0 - abs(err) / math.pi)
    return {"linear_velocity": float(linear), "angular_velocity": float(angular)}


def main() -> int:
    """Run the fake planner over stdin requests."""
    if sys.stdin.isatty():
        return 0
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        robot = payload.get("robot", {})
        goal = payload.get("goal", {})
        if not isinstance(robot, dict) or not isinstance(goal, dict):
            continue
        sys.stdout.write(json.dumps(_command(robot, goal)) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
