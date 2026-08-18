"""Compatibility import for :mod:`robot_sf.benchmark.map_runner.map_runner_worker`."""

from __future__ import annotations

import sys as _sys

from robot_sf.benchmark.map_runner import map_runner_worker as _implementation

_sys.modules[__name__] = _implementation
