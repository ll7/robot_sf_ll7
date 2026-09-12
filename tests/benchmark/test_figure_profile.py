"""Final-size rendering-profile contract tests.

The synthetic records below are presentation fixtures only, not benchmark evidence.
"""

from __future__ import annotations

import json

import pytest

from robot_sf.benchmark.figures.profile import SCHEMA, FigureProfile
from robot_sf.benchmark.figures.scenario_pack import PackConfig, render_view
from robot_sf.benchmark.figures.style import metric_label


def dissertation_profile() -> FigureProfile:
    """Return a compact test profile at the dissertation's 150 mm target width."""
    return FigureProfile(
        profile_id="dissertation-full-width",
        target_width_in=150.0 / 25.4,
        height_ratio=0.8,
        font_family=("DejaVu Serif", "serif"),
        language="en",
        font_size_pt=10.0,
        axes_label_size_pt=10.0,
        axes_title_size_pt=10.0,
        legend_size_pt=8.5,
        tick_label_size_pt=9.0,
        annotation_size_pt=8.0,
        line_width_pt=1.5,
        marker_size_pt=4.0,
        dpi=300,
    )


def prepared_series() -> dict:
    """Return the minimal prepared case needed by a time-series view."""
    return {
        "case": {
            "case_id": "fixture",
            "scenario_id": "synthetic-crossing",
            "planner": "fixture",
            "seed": 7,
        },
        "trace": {"steps": []},
        "times": [0.0, 0.5, 1.0],
        "series": {
            "clearance": [1.0, 0.8, 0.6],
            "speed": [0.5, 0.6, 0.7],
            "turn": [0.0, 0.1, 0.0],
        },
        "event_times": [0.5],
    }


def test_profile_round_trip_has_stable_normalized_digest(tmp_path):
    profile = dissertation_profile()
    path = tmp_path / "profile.json"
    path.write_text(profile.canonical_json(), encoding="utf-8")

    loaded = FigureProfile.from_file(path)

    assert loaded == profile
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == SCHEMA
    assert loaded.sha256() == profile.sha256()
    assert len(profile.sha256()) == 64


def test_profile_rejects_unknown_duplicate_incomplete_and_implausible_fields(tmp_path):
    payload = dissertation_profile().payload()

    payload["unknown"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        FigureProfile.from_file(path)

    path.write_text(
        '{"schema_version":"robot-sf-figure-profile.v1","profile_id":"a",'
        '"profile_id":"b","font_family":["serif"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        FigureProfile.from_file(path)

    path.write_text(
        json.dumps({"schema_version": SCHEMA, "font_family": ["serif"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        FigureProfile.from_file(path)

    with pytest.raises(ValueError, match="target_width"):
        FigureProfile(
            profile_id="bad-width",
            target_width_in=0.0,
            height_ratio=0.8,
            font_family=("serif",),
        )


def test_builtin_profiles_preserve_existing_scenario_pack_dimensions():
    assert FigureProfile.builtin("single").figure_size() == pytest.approx((3.4, 6.2))
    assert FigureProfile.builtin("double").figure_size() == pytest.approx((7.0, 5.6))


def test_time_series_uses_exact_final_size_and_shared_metric_labels():
    profile = dissertation_profile()
    config = PackConfig(mode="diagnostic", formats=("svg",), views=("speed",))

    figure, receipt = render_view(prepared_series(), "speed", config, profile=profile)
    try:
        figure.canvas.draw()
        assert tuple(figure.get_size_inches()) == pytest.approx(profile.figure_size())
        axis = figure.axes[0]
        assert axis.get_xlabel() == metric_label("recorded_time")
        assert axis.get_ylabel() == metric_label("applied_linear_speed")
        assert axis.xaxis.label.get_fontsize() == pytest.approx(profile.axes_label_size_pt)
        assert axis.yaxis.label.get_fontsize() == pytest.approx(profile.axes_label_size_pt)
        assert {tick.get_fontsize() for tick in axis.get_xticklabels()} == {
            profile.tick_label_size_pt
        }
        assert receipt["figure_profile_id"] == profile.profile_id
        assert receipt["figure_profile_sha256"] == profile.sha256()
        assert receipt["figure_size_in"] == pytest.approx(list(profile.figure_size()))
    finally:
        figure.clear()


def test_profile_reports_the_resolved_family_without_machine_paths():
    resolved = dissertation_profile().resolve_font_family()
    assert resolved
    assert "/" not in resolved
    assert "\\" not in resolved
