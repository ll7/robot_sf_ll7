"""Compatibility import for the canonical map-runner policy actions module."""

from __future__ import annotations

import sys

from robot_sf.benchmark.map_runner_policies import map_runner_policy_actions as _canonical
from robot_sf.benchmark.map_runner_policies.map_runner_policy_actions import (
    ppo_action_to_unicycle,
    update_adapter_impact_metrics,
)

__all__ = ("ppo_action_to_unicycle", "update_adapter_impact_metrics")

sys.modules[__name__] = _canonical
