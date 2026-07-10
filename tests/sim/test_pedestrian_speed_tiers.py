"""Tests for the pedestrian speed-tier axis (issue #4972)."""

import pytest

from robot_sf.sim.pedestrian_speed_tiers import (
    PED_SPEED_TIER_BRISK,
    PED_SPEED_TIER_SLOW,
    PED_SPEED_TIER_TYPICAL,
    desired_speed_params_for_tier,
    normalize_ped_speed_tier,
)


def test_tier_mapping_matches_literature_calibration():
    """slow mirrors the legacy default; typical matches Moussaïd et al. 2010."""
    assert desired_speed_params_for_tier(PED_SPEED_TIER_SLOW) == (0.65, pytest.approx(0.2))
    assert desired_speed_params_for_tier(PED_SPEED_TIER_TYPICAL) == (1.3, pytest.approx(0.2))
    assert desired_speed_params_for_tier(PED_SPEED_TIER_BRISK) == (1.6, pytest.approx(0.2))


def test_none_tier_yields_legacy_default_params():
    """A None tier must signal the legacy spawn-coupled derivation."""
    assert desired_speed_params_for_tier(None) == (None, None)


def test_normalize_ped_speed_tier_accepts_known_and_none():
    """Known tiers normalize to canonical lowercase keys; None passes through."""
    assert normalize_ped_speed_tier(None) is None
    assert normalize_ped_speed_tier("Typical") == "typical"
    assert normalize_ped_speed_tier("  Brisk  ") == "brisk"


def test_normalize_ped_speed_tier_rejects_unknown():
    """Unknown tier values must raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unsupported ped_speed_tier"):
        normalize_ped_speed_tier("sprint")
