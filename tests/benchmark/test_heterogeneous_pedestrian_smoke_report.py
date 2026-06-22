"""Tests for issue #3206 heterogeneous-pedestrian smoke reporting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_builder() -> ModuleType:
    module_path = REPO_ROOT / "scripts/benchmark/build_heterogeneous_pedestrian_smoke_report.py"
    spec = importlib.util.spec_from_file_location(
        "build_heterogeneous_pedestrian_smoke_report", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_heterogeneous_pedestrian_smoke_report"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _row(condition: str, seed: int, min_distance: float, support: int = 0) -> dict[str, object]:
    return {
        "scenario_id": f"scenario_{condition}",
        "seed": seed,
        "git_hash": "abc123",
        "scenario_params": {
            "metadata": {"archetype_condition": condition},
            "simulation_config": {
                "archetype_composition": (
                    {"standard": 1.0}
                    if condition == "homogeneous_standard"
                    else {"cautious": 0.34, "standard": 0.33, "hurried": 0.33}
                ),
                "archetype_speed_factors": {
                    "cautious": 0.7,
                    "standard": 1.0,
                    "hurried": 1.4,
                },
                "archetype_seed": 3206,
            },
        },
        "metrics": {
            "success": False,
            "collisions": 0,
            "min_distance": min_distance,
            "mean_distance": min_distance + 1.0,
            "robot_ped_within_5m_frac": 0.5,
            "distributional_disruption": {
                "support_counts": {
                    "slow_speed_tier": support,
                    "fast_speed_tier": 0,
                    "extreme_speed_tier": 0,
                },
                "missing_data": {
                    "slow_speed_tier": {
                        "status": "unavailable",
                        "reason": "No control trace provided",
                    }
                },
            },
        },
    }


def test_smoke_report_records_metric_delta_and_unavailable_distributional_support() -> None:
    """The report should keep deltas separate from fairness-readiness claims."""
    rows = [
        _row("homogeneous_standard", 101, 3.0),
        _row("homogeneous_standard", 102, 5.0),
        _row("mixed_balanced", 101, 7.0),
        _row("mixed_balanced", 102, 9.0),
    ]

    report = builder.build_report(rows)

    assert report["status"] == "diagnostic_smoke_report"
    assert report["delta_variant_minus_baseline"]["min_distance"][
        "absolute_delta"
    ] == pytest.approx(4.0)
    assert (
        report["conditions"]["mixed_balanced"]["distributional_disruption"]["status"]
        == "not_computable"
    )
    assert report["per_archetype_distributional_status"] == "not_computable_from_current_smoke"


def test_smoke_report_marks_distributional_status_computable_when_support_exists() -> None:
    """Positive distributional support counts should not be hidden as unavailable."""
    rows = [
        _row("homogeneous_standard", 101, 3.0, support=2),
        _row("mixed_balanced", 101, 4.0, support=3),
    ]

    report = builder.build_report(rows)

    assert (
        report["conditions"]["homogeneous_standard"]["distributional_disruption"]["status"]
        == "computable"
    )
    assert report["per_archetype_distributional_status"] == "computable"


def test_smoke_report_markdown_records_issue_3261_scope_decision() -> None:
    """Generated evidence README text should preserve the #3261 decision boundary."""
    report = builder.build_report(
        [
            _row("homogeneous_standard", 101, 3.0),
            _row("mixed_balanced", 101, 4.0),
        ]
    )

    markdown = builder.format_markdown(report)

    assert "## Issue #3261 Archetype Interpretation (2026-06-22)" in markdown
    assert "`slow_speed_tier`, `fast_speed_tier`, and `extreme_speed_tier`" in markdown
    assert "`per_archetype_distributional_status` remains" in markdown
    assert "`not_computable_from_current_smoke`" in markdown


def test_smoke_report_requires_baseline_and_variant_conditions() -> None:
    """Missing comparison conditions should fail loudly."""
    with pytest.raises(ValueError, match="variant condition"):
        builder.build_report([_row("homogeneous_standard", 101, 3.0)])
