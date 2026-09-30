"""Versioned waypoint selection for planners with a historical next-goal shortcut."""

from __future__ import annotations

import numpy as np

LEGACY_NEXT_GOAL_V1 = "legacy_next_goal_v1"
ACTIVE_WAYPOINT_V2 = "active_waypoint_v2"


def validate_goal_target_version(version: str) -> None:
    """Reject unknown selectors before an evaluation starts."""
    if version not in (LEGACY_NEXT_GOAL_V1, ACTIVE_WAYPOINT_V2):
        raise ValueError(f"Unsupported goal_target_version: {version!r}")


def select_goal_target(
    robot_pos: np.ndarray,
    goal_current: np.ndarray,
    goal_next: np.ndarray,
    *,
    version: str,
) -> np.ndarray:
    """Select the planner target under the requested route contract.

    The observation producer uses ``[0, 0]`` for an absent next waypoint.
    In v2, the simulator owns waypoint advancement and ``goal.current`` is
    always the active target, including when the route legitimately ends at
    the map origin. V1 intentionally retains the historical selection rule.

    Returns:
        np.ndarray: Selected world-frame goal position.
    """
    validate_goal_target_version(version)
    if version == ACTIVE_WAYPOINT_V2:
        return goal_current
    return goal_next if np.linalg.norm(goal_next - robot_pos) > 1e-6 else goal_current
