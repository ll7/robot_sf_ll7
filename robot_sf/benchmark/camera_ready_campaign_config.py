"""Backward-compatible config dataclass import surface.

The definitions now live in ``robot_sf.benchmark.camera_ready._config_types`` as
part of the #3385 camera-ready package decomposition. Keep this module as a
compatibility facade for callers that imported the earlier extraction module
directly.
"""

from robot_sf.benchmark.camera_ready._config_types import (  # noqa: F401
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
