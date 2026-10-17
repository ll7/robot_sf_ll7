"""Offline admission of documented undefined metric sentinels (issue #10016 F3)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from robot_sf.analysis_workbench.audit_detectors import detect, normalize_recorded_undefined
from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.benchmark.metrics import (
    EpisodeData,
    comfort_exposure,
    force_quantiles,
    min_distance,
    time_to_goal,
)

# Names whose metric definitions explicitly permit undefined values, not arbitrary
# distances/forces, flags, counts, speeds, telemetry, or caller-authored metadata.
UNDEFINED_SCALARS = (
    "time_to_goal",
    "time_to_goal_norm_success_only",
    "time_to_goal_ideal_ratio",
    "path_efficiency",
    "aggregated_time",
    "min_distance",
    "mean_distance",
    "min_clearance",
    "mean_clearance",
    "robot_ped_within_5m_frac",
    "distance_to_human_min",
    "clearing_distance_min",
    "clearing_distance_avg",
    "time_to_collision_min",
    "force_q50",
    "force_q90",
    "force_q95",
    "ped_force_q50",
    "ped_force_q90",
    "ped_force_q95",
    "ped_force_mean",
    "force_exceed_events",
    "comfort_exposure",
    "force_gradient_norm_mean",
)


@pytest.mark.parametrize("container", ["metrics", "operational_metrics"])
@pytest.mark.parametrize("name", UNDEFINED_SCALARS)
def test_documented_nan_is_missing_with_path_and_unchanged_source(container, name):
    row = {"episode_id": "undefined", container: {"clearance_m": 1.0, name: float("nan")}}
    projected = normalize_recorded_undefined(row)
    assert projected[container][name] is None
    assert math.isnan(row[container][name])
    assert projected["audit_adapter_missingness"] == [
        {
            "path": f"{container}.{name}",
            "source_token": "NaN",
            "reason": "documented_undefined_metric",
        }
    ]
    json.dumps(projected, allow_nan=False)
    assert detect("extreme_measurements", row).status == "clear"


@pytest.mark.parametrize("container", ["metrics", "operational_metrics", None])
def test_quantile_nan_is_missing_at_supported_structured_aliases(container):
    record = {"force_quantiles": {"q50": float("nan"), "q90": 1.0, "q95": 2.0}}
    row = {"episode_id": "quantiles", "metrics": {"clearance_m": 1.0}}
    if container:
        row.setdefault(container, {}).update(record)
    else:
        row.update(record)
    projected = normalize_recorded_undefined(row)
    source = projected[container] if container else projected
    assert source["force_quantiles"] == {"q50": None, "q90": 1.0, "q95": 2.0}
    assert detect("extreme_measurements", row).status == "clear"


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), True, "NaN", [], {}])
@pytest.mark.parametrize("name", ["path_efficiency", "force_q50"])
def test_other_invalid_values_at_allowlisted_names_remain_errors(name, value):
    row = {"episode_id": "invalid", "metrics": {name: value, "clearance_m": 1.0}}
    assert normalize_recorded_undefined(row) is row
    assert detect("extreme_measurements", row).status == "error"


@pytest.mark.parametrize("name", ["avg_speed", "collisions", "clearance_m", "made_up_force"])
def test_nan_outside_documented_names_remains_error(name):
    row = {"episode_id": "invalid", "metrics": {name: float("nan")}}
    assert normalize_recorded_undefined(row) is row
    assert detect("extreme_measurements", row).status == "error"


def test_matching_name_in_metadata_or_unknown_structured_path_remains_error():
    for extra in (
        {"metadata": {"path_efficiency": float("nan")}},
        {"metrics": {"extension": {"path_efficiency": float("nan")}}},
        {"metric_values": {"path_efficiency": float("nan")}},
    ):
        row = {"episode_id": "invalid", "metrics": {"clearance_m": 1.0}, **extra}
        assert normalize_recorded_undefined(row) is row
        assert detect("extreme_measurements", row).status == "error"


def test_production_undefined_metrics_survive_serialization_and_scan(tmp_path):
    """Use raw producer output, rather than a pre-normalized fixture."""
    data = EpisodeData(
        robot_pos=np.zeros((3, 2)),
        robot_vel=np.zeros((3, 2)),
        robot_acc=np.zeros((3, 2)),
        peds_pos=np.empty((3, 0, 2)),
        ped_forces=np.empty((3, 0, 2)),
        goal=np.array([5.0, 0.0]),
        dt=0.1,
    )
    raw = {
        "time_to_goal": time_to_goal(data),
        "min_distance": min_distance(data),
        "comfort_exposure": comfort_exposure(data),
        **force_quantiles(data),
    }
    assert math.isnan(raw["time_to_goal"])
    assert math.isnan(raw["min_distance"])
    assert math.isnan(raw["force_q50"])
    row = {"episode_id": "produced", "metrics": {"clearance_m": 1.0, **raw}}
    source = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    original = source.read_bytes()
    report = scan_campaign(source, root=tmp_path, detector_ids=["extreme_measurements"])
    assert report.inventory[0].status == "readable"
    assert report.signals[0].status == "clear"
    assert source.read_bytes() == original
