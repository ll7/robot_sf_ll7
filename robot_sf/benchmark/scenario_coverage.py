"""Compatibility import for :mod:`robot_sf.benchmark.scenario.scenario_coverage`.

The scenario package owns the implementation; this module alias preserves the historical import
path and module identity for callers that still use it.
"""

from __future__ import annotations

import sys as _sys

from robot_sf.benchmark.scenario import scenario_coverage as _implementation

_sys.modules[__name__] = _implementation
