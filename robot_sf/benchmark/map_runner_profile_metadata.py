"""Compatibility import for the canonical map-runner profile metadata helpers."""

from __future__ import annotations

import sys

from robot_sf.benchmark.map_runner_policies import map_runner_profile_metadata as _canonical
from robot_sf.benchmark.map_runner_policies.map_runner_profile_metadata import (
    load_latency_profile,
    load_synthetic_actuation_profile,
)

__all__ = ("load_latency_profile", "load_synthetic_actuation_profile")

sys.modules[__name__] = _canonical
