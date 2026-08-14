"""Compatibility import for :mod:`robot_sf.benchmark.campaign.campaign_checkpoint_preflight`.

The module alias keeps legacy imports and monkeypatch-sensitive callers on the same module object
as the canonical campaign package implementation.
"""

from __future__ import annotations

import sys as _sys

from robot_sf.benchmark.campaign import campaign_checkpoint_preflight as _implementation

_sys.modules[__name__] = _implementation
