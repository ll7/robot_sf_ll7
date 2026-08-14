"""Import and compatibility contracts for the benchmark campaign package."""

from __future__ import annotations

import importlib

import pytest

from robot_sf.benchmark import campaign

_MODULE_NAMES = (
    "campaign_arm_admission",
    "campaign_atlas",
    "campaign_checkpoint_preflight",
    "campaign_logging",
    "campaign_runtime_preflight",
)


def test_campaign_package_exposes_only_lazy_module_names() -> None:
    """Keep the package initializer dependency-light and explicit."""

    assert tuple(campaign.__all__) == _MODULE_NAMES


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_legacy_campaign_module_aliases_canonical_module(module_name: str) -> None:
    """Preserve old imports and monkeypatch identity while exposing canonical paths."""

    canonical = importlib.import_module(f"robot_sf.benchmark.campaign.{module_name}")
    legacy = importlib.import_module(f"robot_sf.benchmark.{module_name}")

    assert legacy is canonical
    assert legacy.__name__ == f"robot_sf.benchmark.campaign.{module_name}"
