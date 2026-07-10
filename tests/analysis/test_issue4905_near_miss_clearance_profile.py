"""Regression tests for the Issue #4905 near-miss clearance profile."""

from __future__ import annotations

import pandas as pd

from scripts.analysis import issue4905_near_miss_clearance_profile as profile


def test_occlusion_share_is_unavailable_when_visibility_is_disabled() -> None:
    """All missing visibility observations must not be reported as a zero share."""
    frame = pd.DataFrame(
        [
            {
                "planner": "orca",
                "actual_minimum_separation_m": 0.4,
                "was_occluded_before_min": None,
            },
            {
                "planner": "orca",
                "actual_minimum_separation_m": 0.3,
                "was_occluded_before_min": None,
            },
        ]
    )

    result = profile.compute_occlusion_share(frame)

    assert result.loc[0, "occlusion_measurement"] == "unavailable"
    assert pd.isna(result.loc[0, "occluded_share"])
    assert result.loc[0, "occluded_none"] == 2


def test_occlusion_share_remains_numeric_when_visibility_is_observed() -> None:
    """Observed occlusion values retain the established per-planner share."""
    frame = pd.DataFrame(
        [
            {
                "planner": "orca",
                "actual_minimum_separation_m": 0.4,
                "was_occluded_before_min": True,
            },
            {
                "planner": "orca",
                "actual_minimum_separation_m": 0.3,
                "was_occluded_before_min": False,
            },
        ]
    )

    result = profile.compute_occlusion_share(frame)

    assert result.loc[0, "occlusion_measurement"] == "available"
    assert result.loc[0, "occluded_share"] == 0.5


def test_occlusion_share_ignores_missing_records_when_visibility_is_mixed() -> None:
    """An observed occlusion share uses only records with a visibility observation."""
    frame = pd.DataFrame(
        [
            {
                "planner": "orca",
                "actual_minimum_separation_m": 0.4,
                "was_occluded_before_min": True,
            },
            {
                "planner": "orca",
                "actual_minimum_separation_m": 0.3,
                "was_occluded_before_min": None,
            },
        ]
    )

    result = profile.compute_occlusion_share(frame)

    assert result.loc[0, "occlusion_measurement"] == "available"
    assert result.loc[0, "occluded_share"] == 1.0
    assert result.loc[0, "occluded_none"] == 1
