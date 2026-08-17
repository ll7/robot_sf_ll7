"""Compatibility import for the canonical map-runner policy common module."""

from __future__ import annotations

import sys

from robot_sf.benchmark.map_runner_policies import map_runner_policy_common as _canonical
from robot_sf.benchmark.map_runner_policies.map_runner_policy_common import build_adapter_policy

__all__ = ("build_adapter_policy",)

sys.modules[__name__] = _canonical
