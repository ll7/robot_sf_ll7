"""Compatibility import for the canonical map-runner policy actions module."""

from __future__ import annotations

import sys

from robot_sf.benchmark.map_runner_policies import map_runner_policy_actions as _canonical

sys.modules[__name__] = _canonical
