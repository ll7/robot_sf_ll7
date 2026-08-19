"""Compatibility import for :mod:`robot_sf.benchmark.map_runner.map_runner_view_integrity`.

The map runner package owns the implementation; this module alias preserves the historical
import path and module identity for callers that still use it.
"""

from __future__ import annotations

import sys as _sys

from robot_sf.benchmark.map_runner import map_runner_view_integrity as _implementation

_sys.modules[__name__] = _implementation
