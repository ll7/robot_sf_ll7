"""Compatibility alias for the camera-ready campaign package facade.

The implementation lives in ``robot_sf.benchmark.camera_ready._legacy_campaign_facade`` so the
``camera_ready`` package owns the remaining compatibility adapter. Replacing this module entry keeps
legacy imports and monkeypatches against ``robot_sf.benchmark.camera_ready_campaign`` pointed at the
same adapter module.
"""

from __future__ import annotations

import sys

from robot_sf.benchmark.camera_ready import _legacy_campaign_facade as _facade

sys.modules[__name__] = _facade
