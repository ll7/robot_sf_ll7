"""Compatibility import for the canonical map-runner policy metadata module."""

from __future__ import annotations

import sys

from robot_sf.benchmark.map_runner_policies import map_runner_policy_metadata as _canonical
from robot_sf.benchmark.map_runner_policies.map_runner_policy_metadata import (
    apply_direct_world_velocity_metadata,
    attach_planner_reset,
    finalize_feasibility_metadata,
    holonomic_world_velocity_command,
)

__all__ = (
    "apply_direct_world_velocity_metadata",
    "attach_planner_reset",
    "finalize_feasibility_metadata",
    "holonomic_world_velocity_command",
)

sys.modules[__name__] = _canonical
