"""Backward-compatible dataclass surface.

The definitions now live under ``robot_sf.benchmark.camera_ready`` as part of
the #3385 camera-ready package decomposition. Keep this module as a compatibility
facade for callers that imported the earlier extraction module directly.
"""

from robot_sf.benchmark.camera_ready import (  # noqa: F401
    _AMV_DIMENSIONS,
    DEFAULT_SEED_SETS_PATH,
    AmvProfileConfig,
    CampaignConfig,
    PlannerSpec,
    ScenarioCandidateSelection,
    SeedPolicy,
    SnqiContractConfig,
)

__all__ = [
    "DEFAULT_SEED_SETS_PATH",
    "AmvProfileConfig",
    "CampaignConfig",
    "PlannerSpec",
    "ScenarioCandidateSelection",
    "SeedPolicy",
    "SnqiContractConfig",
]
