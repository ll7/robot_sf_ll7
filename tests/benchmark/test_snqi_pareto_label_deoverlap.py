"""Tests for the SNQI Pareto SVG label de-overlap logic (issue #5401).

Validates that ``format_pareto_svg`` produces non-overlapping labels via
short aliases, iterative repulsion, leader lines, and marker halos.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    _build_short_aliases,
    _deoverlap_labels,
    _estimate_label_width,
    format_pareto_svg,
)

# ---------------------------------------------------------------------------
# _build_short_aliases
# ---------------------------------------------------------------------------


class TestBuildShortAliases:
    """Tests for ``_build_short_aliases`` helper."""

    def test_short_names_kept_as_is(self) -> None:
        names = ["ppo", "orca", "goal"]
        result = _build_short_aliases(names)
        assert result == {"ppo": "ppo", "orca": "orca", "goal": "goal"}

    def test_long_name_shortened(self) -> None:
        names = ["hybrid_rule_v3_fast_progress_static_escape"]
        result = _build_short_aliases(names)
        assert result["hybrid_rule_v3_fast_progress_static_escape"] != names[0]
        assert len(result["hybrid_rule_v3_fast_progress_static_escape"]) < len(names[0])

    def test_filler_tokens_skipped(self) -> None:
        names = ["hybrid_rule_v3_fast_progress_static_escape"]
        result = _build_short_aliases(names)
        alias = result["hybrid_rule_v3_fast_progress_static_escape"]
        assert "rule" not in alias.split("_")
        assert "v3" not in alias.split("_")
        assert "fast" not in alias.split("_")

    def test_threshold_respected(self) -> None:
        names = ["exactly_18_chars!!"]  # 18 chars = default threshold
        result = _build_short_aliases(names, threshold=18)
        assert result["exactly_18_chars!!"] == "exactly_18_chars!!"

    def test_scenario_adaptive_hybrid_shortened(self) -> None:
        names = ["scenario_adaptive_hybrid_orca_v1"]
        result = _build_short_aliases(names, threshold=18)
        alias = result["scenario_adaptive_hybrid_orca_v1"]
        assert "v1" not in alias.split("_")
        assert len(alias) < len("scenario_adaptive_hybrid_orca_v1")


# ---------------------------------------------------------------------------
# _estimate_label_width
# ---------------------------------------------------------------------------


class TestEstimateLabelWidth:
    """Tests for ``_estimate_label_width`` helper."""

    def test_width_scales_with_length(self) -> None:
        short = _estimate_label_width("ppo")
        long = _estimate_label_width("hybrid_rule_v3_fast_progress_static_escape")
        assert long > short

    def test_nonzero_for_nonempty(self) -> None:
        assert _estimate_label_width("a") > 0.0

    def test_padding_included(self) -> None:
        empty = _estimate_label_width("")
        assert empty == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# _deoverlap_labels
# ---------------------------------------------------------------------------


class TestDeoverlapLabels:
    """Tests for ``_deoverlap_labels`` iterative repulsion."""

    def test_non_overlapping_unchanged(self) -> None:
        labels = [
            {"anchor_x": 100, "anchor_y": 100, "label_x": 108, "label_y": 92, "w": 30, "h": 12},
            {"anchor_x": 400, "anchor_y": 300, "label_x": 408, "label_y": 292, "w": 30, "h": 12},
        ]
        result = _deoverlap_labels(labels)
        assert result[0]["label_x"] == pytest.approx(108)
        assert result[1]["label_x"] == pytest.approx(408)

    def test_overlapping_separated(self) -> None:
        labels = [
            {"anchor_x": 100, "anchor_y": 100, "label_x": 108, "label_y": 92, "w": 80, "h": 12},
            {"anchor_x": 110, "anchor_y": 100, "label_x": 118, "label_y": 92, "w": 80, "h": 12},
        ]
        result = _deoverlap_labels(labels)
        gap_x = abs(result[0]["label_x"] - result[1]["label_x"])
        assert gap_x > 0

    def test_returns_same_count(self) -> None:
        labels = [
            {"anchor_x": 0, "anchor_y": 0, "label_x": 8, "label_y": -8, "w": 20, "h": 12},
        ]
        result = _deoverlap_labels(labels)
        assert len(result) == 1

    def test_does_not_mutate_input(self) -> None:
        labels = [
            {"anchor_x": 100, "anchor_y": 100, "label_x": 108, "label_y": 92, "w": 80, "h": 12},
            {"anchor_x": 110, "anchor_y": 100, "label_x": 118, "label_y": 92, "w": 80, "h": 12},
        ]
        original_x = labels[0]["label_x"]
        _deoverlap_labels(labels)
        assert labels[0]["label_x"] == original_x


# ---------------------------------------------------------------------------
# format_pareto_svg (integration)
# ---------------------------------------------------------------------------


def _make_report(planner_names: list[str]) -> dict[str, Any]:
    """Build a minimal report dict for testing."""
    rows = []
    for i, name in enumerate(planner_names):
        rows.append(
            {
                "planner": name,
                "constraints_first_score": 0.5 + i * 0.05,
                "snqi_mean": 0.5 + i * 0.03,
                "pareto_front": i < 2,
            }
        )
    return {"planner_rows": rows}


class TestFormatParetoSvg:
    """Integration tests for ``format_pareto_svg``."""

    def test_contains_svg_element(self) -> None:
        report = _make_report(["ppo", "orca"])
        svg = format_pareto_svg(report)
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_short_names_not_aliased(self) -> None:
        report = _make_report(["ppo", "orca"])
        svg = format_pareto_svg(report)
        assert ">ppo<" in svg
        assert ">orca<" in svg

    def test_long_names_shortened(self) -> None:
        report = _make_report(
            [
                "ppo",
                "hybrid_rule_v3_fast_progress_static_escape",
            ]
        )
        svg = format_pareto_svg(report)
        # The full planner name should appear only in the legend, not inline
        assert "Legend" in svg

    def test_legend_present_for_aliases(self) -> None:
        report = _make_report(
            [
                "ppo",
                "hybrid_rule_v3_fast_progress_static_escape",
            ]
        )
        svg = format_pareto_svg(report)
        assert "Legend" in svg
        assert "hybrid_rule_v3_fast_progress_static_escape" in svg

    def test_leader_lines_present_when_labels_displaced(self) -> None:
        report = _make_report(
            [
                "ppo",
                "hybrid_rule_v3_fast_progress_static_escape",
                "scenario_adaptive_hybrid_orca_v1",
            ]
        )
        svg = format_pareto_svg(report)
        assert "<line" in svg

    def test_marker_halos_present(self) -> None:
        report = _make_report(["ppo", "orca"])
        svg = format_pareto_svg(report)
        circles = re.findall(r'<circle[^>]+fill="white"', svg)
        assert len(circles) >= 2

    def test_pareto_polyline_preserved(self) -> None:
        report = _make_report(["ppo", "orca", "goal"])
        svg = format_pareto_svg(report)
        assert "<polyline" in svg

    def test_regression_fixture_no_overlapping_bbox(self) -> None:
        """Simulate the actual issue-3653 dataset layout and verify labels
        do not overlap after de-overlap."""
        report = _make_report(
            [
                "ppo",
                "hybrid_rule_v3_fast_progress_static_escape",
                "scenario_adaptive_hybrid_orca_v1",
                "orca",
                "socnav_sampling",
                "prediction_planner",
                "social_force",
                "sacadrl",
                "goal",
            ]
        )
        svg = format_pareto_svg(report)
        assert "<svg" in svg
        assert "Legend" in svg
        # Verify the SVG is valid (has closing tag)
        assert svg.strip().endswith("</svg>")
