"""Versioned metric/planner display-semantics contract tests."""

from __future__ import annotations

import json

import pytest

from robot_sf.benchmark.figures.scenario_pack import PackConfig, render_view
from robot_sf.benchmark.figures.semantics import (
    SCHEMA,
    SemanticsRegistry,
    default_registry,
)
from robot_sf.benchmark.figures.style import metric_label, semantics_sha256


def prepared_series() -> dict:
    """Return the minimal prepared record for a time-series render."""
    return {
        "case": {
            "case_id": "fixture",
            "scenario_id": "synthetic-crossing",
            "planner": "orca",
            "seed": 7,
        },
        "trace": {"steps": []},
        "times": [0.0, 0.5, 1.0],
        "series": {
            "clearance": [1.0, 0.8, 0.6],
            "speed": [0.5, 0.6, 0.7],
            "turn": [0.0, 0.1, 0.0],
        },
        "event_times": [],
    }


def test_packaged_registry_is_complete_stable_and_alias_aware():
    registry = default_registry()

    assert len(registry.metrics) == 35
    assert len(registry.planners) == 8
    assert len(registry.sha256()) == 64
    assert semantics_sha256() == registry.sha256()
    assert registry.metric("surface_clearance").key == "min_clearance"
    assert registry.metric_label("surface_clearance") == (
        "Minimum robot-pedestrian surface clearance (m)"
    )
    assert registry.metric_label("surface_clearance", language="de", short=True) == (
        "Minimalabstand (m)"
    )
    assert registry.planner("ORCA").key == "orca"
    assert registry.planner_label("ORCA") == "ORCA"


def test_style_compatibility_is_explicit_while_strict_api_rejects_unknowns():
    assert metric_label("success") == "Success rate"
    assert metric_label("collisions") == "Collision rate"
    assert metric_label("unknown_metric") == "Unknown Metric"
    assert metric_label("") == "Metric"
    with pytest.raises(KeyError, match="unmapped metric"):
        metric_label("unknown_metric", strict=True)
    with pytest.raises(KeyError, match="unmapped planner"):
        default_registry().planner_label("unknown_planner")


def test_profile_language_selects_registry_labels():
    from robot_sf.benchmark.figures.profile import FigureProfile

    profile = FigureProfile(
        profile_id="german-review",
        target_width_in=150.0 / 25.4,
        height_ratio=0.8,
        font_family=("DejaVu Serif", "serif"),
        language="de",
    )
    config = PackConfig(mode="diagnostic", formats=("svg",), views=("speed",))

    figure, _receipt = render_view(prepared_series(), "speed", config, profile=profile)
    try:
        assert figure.axes[0].get_xlabel() == "Absolute aufgezeichnete Zeit (s)"
        assert figure.axes[0].get_ylabel() == "Aufgezeichnete Stellgeschwindigkeit (m/s)"
    finally:
        figure.clear()


def test_registry_rejects_alias_collisions_and_unknown_fields(tmp_path):
    payload = default_registry().payload()
    payload["metrics"]["time_to_goal"]["aliases"].append("success")
    path = tmp_path / "collision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="maps to both"):
        SemanticsRegistry.from_file(path)

    payload = default_registry().payload()
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level"):
        SemanticsRegistry.from_file(path)


def test_suggestion_is_review_only_and_does_not_mutate_registry():
    registry = default_registry()
    before = registry.sha256()

    proposal = registry.suggest_metric(
        "lateral_jerk",
        unit="m/s³",
        contexts=("robot_sf/example.py:42", "docs/figure-brief.md"),
    )

    assert proposal["proposal_only"] is True
    assert proposal["registry_schema"] == SCHEMA
    assert proposal["candidate"]["labels"]["de"] == "REVIEW REQUIRED"
    assert proposal["contexts"] == ["robot_sf/example.py:42", "docs/figure-brief.md"]
    assert registry.sha256() == before
    with pytest.raises(ValueError, match="already mapped"):
        registry.suggest_metric("success")


def test_planner_rows_always_include_non_color_distinction():
    registry = default_registry()
    combinations = {(row.marker, row.line_style) for row in registry.planners.values()}

    assert all(row.marker and row.line_style for row in registry.planners.values())
    assert len(combinations) >= 6
