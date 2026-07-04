"""Camera-ready campaign package.

Decomposes historically monolithic ``camera_ready_campaign.py`` (~4967 LOC)
into focused submodules (see #3385). Leaf helpers live in ``_util.py``; config
dataclasses live in ``_config_types.py`` and are re-exported here as package
surface while ``camera_ready_campaign_config.py`` remains compatibility facade.
Most orchestration/IO/computation moved into package while
``camera_ready_campaign.py`` preserves legacy import monkeypatch surfaces.
"""

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.camera_ready._config_types import _AMV_DIMENSIONS as _AMV_DIMENSIONS
from robot_sf.benchmark.camera_ready._config_types import (
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
    "load_campaign_config",
]
