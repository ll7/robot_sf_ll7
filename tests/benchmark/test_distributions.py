"""Tests for benchmark distribution plotting utilities."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pytest

from robot_sf.benchmark.distributions import collect_grouped_values, save_distributions
from robot_sf.benchmark.errors import AggregationMetadataError


def test_collect_grouped_values_filters_invalid_entries() -> None:
    """Ensure grouped value collection skips invalid or missing metrics."""
    records = [
        {"scenario_id": "s1", "metrics": {"success": 1.0}},
        {"scenario_params": {"algo": "a"}, "metrics": {"success": 0.5, "loss": float("nan")}},
        {"scenario_params": {"algo": "a"}, "metrics": {"success": "bad"}},
    ]
    grouped = collect_grouped_values(records, metrics=["success", "loss"])
    assert grouped["s1"]["success"] == [1.0]
    assert grouped["a"]["success"] == [0.5]
    assert "loss" not in grouped["a"]


def test_collect_grouped_values_requires_explicit_cross_track_mode() -> None:
    """Distribution inputs should not pool incompatible observation tracks by default."""
    records = [
        {
            "benchmark_track": "grid_socnav_v1",
            "scenario_params": {"algo": "a", "benchmark_track": "grid_socnav_v1"},
            "metrics": {"success": 1.0},
        },
        {
            "benchmark_track": "lidar_2d_v1",
            "scenario_params": {"algo": "a", "benchmark_track": "lidar_2d_v1"},
            "metrics": {"success": 0.0},
        },
    ]

    with pytest.raises(AggregationMetadataError):
        collect_grouped_values(records, metrics=["success"])

    grouped = collect_grouped_values(
        records,
        metrics=["success"],
        observation_track_mode="diagnostic-cross-track",
    )
    assert grouped["grid_socnav_v1 :: a"]["success"] == [1.0]
    assert grouped["lidar_2d_v1 :: a"]["success"] == [0.0]


def test_save_distributions_writes_pngs(tmp_path: Path) -> None:
    """Verify distribution plots are saved to disk for each metric."""
    grouped = {"algoA": {"success_rate": [0.1, 0.2, 0.3, 0.4, 0.5]}}
    meta = save_distributions(grouped, tmp_path, bins=5, kde=False, out_pdf=False, ci=True)
    assert meta.wrote
    assert Path(meta.wrote[0]).exists()
