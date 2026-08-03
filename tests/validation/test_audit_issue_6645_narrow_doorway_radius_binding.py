"""Tests for the issue #6645 narrow-doorway radius-binding audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.narrow_doorway_radius_audit import (
    build_audit_report,
    derive_doorway_geometry,
    envelope_clearance_margin_m,
)
from robot_sf.training.scenario_loader import load_scenarios

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIO_PATH = _REPO_ROOT / "configs/scenarios/single/francis2023_narrow_doorway.yaml"


def _scenario() -> dict:
    return dict(load_scenarios(_SCENARIO_PATH)[0])


@pytest.mark.parametrize(
    ("radius_m", "expected_margin_m"),
    [(0.0, 2.0), (0.5, 1.0), (0.8, 0.4), (1.0, 0.0), (1.1, -0.2)],
)
def test_clearance_boundary_uses_envelope_diameter(
    radius_m: float, expected_margin_m: float
) -> None:
    """The audit preserves the exact gap-minus-diameter boundary, including zero."""
    assert envelope_clearance_margin_m(2.0, radius_m) == pytest.approx(expected_margin_m)


def test_geometry_is_derived_from_authored_map() -> None:
    """The authored SVG yields the 2 m opening and 1 m route center distance."""
    geometry = derive_doorway_geometry(_SCENARIO_PATH, _scenario())

    assert geometry.gap_lower_edge_m == pytest.approx(4.0)
    assert geometry.gap_upper_edge_m == pytest.approx(6.0)
    assert geometry.gap_width_m == pytest.approx(2.0)
    assert geometry.route_min_center_distance_m == pytest.approx(1.0)


def test_full_audit_passes_without_promoting_campaign_evidence() -> None:
    """The real canary passes while the report keeps its diagnostic boundary."""
    report = build_audit_report(_SCENARIO_PATH, radii_m=(0.5, 0.8, 1.0))

    assert report["go"] is True
    assert report["checks"]["zero_clearance_at_1m"] is True
    assert report["checks"]["all_five_surfaces_present"] is True
    assert report["release_or_frozen_artifacts_changed"] is False
    assert report["claim_boundary"].startswith("diagnostic geometry")
    assert report["interpretation"]["positive_margin_does_not_override_grid_oracle_classification"]
