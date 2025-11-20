"""Robot SF package bootstrap and telemetry exports."""

import os
import sys

csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)

from robot_sf.telemetry import (  # noqa: E402 - telemetry depends on package path tweak
    ManifestWriter,
    RunRegistry,
    RunTrackerConfig,
    generate_run_id,
)

__all__ = [
    "ManifestWriter",
    "RunRegistry",
    "RunTrackerConfig",
    "generate_run_id",
]
