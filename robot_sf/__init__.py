"""Robot SF package bootstrap and telemetry exports."""

from . import telemetry as telemetry
from .telemetry import (
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
    "telemetry",
]
