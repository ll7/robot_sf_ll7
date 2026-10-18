"""AUD6 metric-schema metadata and cohort compatibility regressions."""

import hashlib
import json
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.audit_detectors import detect
from robot_sf.analysis_workbench.audit_scan import scan_campaign


@pytest.mark.parametrize("version", ["robot-sf-metrics.v1", "robot-sf-metrics.v2"])
def test_sha_checked_release_rows_accept_metric_schema_metadata(version):
    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures/analysis_workbench/aud_release_0_0_7.json"
    )
    for sample in json.loads(fixture.read_text())["samples"]:
        assert hashlib.sha256(sample["raw_line"].encode()).hexdigest() == sample["line_sha256"]
        row = json.loads(sample["raw_line"])
        row["_source_member"] = sample["member"]
        row["_release_arm"] = sample["member"].split("/")[2].split("__")[0]
        row["metrics"]["metric_schema_version"] = version
        signal = scan_campaign([row], detector_ids=["extreme_measurements"]).signals[0]
        assert signal.status == "clear", (sample["kind"], signal.reason_code)


@pytest.mark.parametrize("value", [None, True, 1, "", "robot-sf-metrics.v3", [], {}])
def test_extreme_measurements_reject_invalid_metric_schema_metadata(value):
    row = {"episode_id": "invalid-schema", "metrics": {"energy": 0, "metric_schema_version": value}}
    signal = detect("extreme_measurements", row)
    assert (signal.status, signal.reason_code) == ("error", "nonfinite_or_malformed_measurement")
    assert "metrics.metric_schema_version" in signal.missingness


def cohort_rows():
    return [
        {
            "episode_id": f"{scenario}-{planner}-{seed}",
            "scenario_id": scenario,
            "planner_id": planner,
            "seed": seed,
            "outcome": {"label": "success"},
            "metrics": {"avg_speed": 0.8, "time_to_goal_norm": 0.5},
        }
        for scenario in ("same", "other")
        for planner in ("target", "peer-a", "peer-b")
        for seed in range(1001, 1007)
    ]


@pytest.mark.parametrize("detector", ["planner_cohort_shift", "outcome_incidence"])
@pytest.mark.parametrize("planner", ["target", "peer-a"])
@pytest.mark.parametrize("location", ["metrics", "row", "both"])
def test_mixed_metric_schema_gates_entire_cell_and_preserves_other_cells(
    detector, planner, location
):
    rows = cohort_rows()
    bad = next(row for row in rows if row["episode_id"] == f"same-{planner}-1006")
    if location in {"metrics", "both"}:
        bad["metrics"]["metric_schema_version"] = "robot-sf-metrics.v2"
    if location in {"row", "both"}:
        bad["metric_schema_version"] = "robot-sf-metrics.v2"
    report = scan_campaign(rows, detector_ids=[detector])
    for signal in report.signals:
        if signal.episode_id.startswith("same-"):
            assert (signal.status, signal.reason_code) == (
                "unavailable",
                "mixed_metric_schema_versions",
            )
            assert signal.measured["metric_schema_versions"] == [
                "robot-sf-metrics.v1",
                "robot-sf-metrics.v2",
            ]
        else:
            assert signal.status == "clear"
    direct = detect(detector, rows[0], cohort=rows)
    assert (direct.status, direct.reason_code) == ("unavailable", "mixed_metric_schema_versions")


@pytest.mark.parametrize("detector", ["planner_cohort_shift", "outcome_incidence"])
@pytest.mark.parametrize("version", [None, "robot-sf-metrics.v1", "robot-sf-metrics.v2"])
def test_homogeneous_metric_schema_cells_remain_clear(detector, version):
    rows = cohort_rows()
    for index, row in enumerate(rows):
        # Explicit v1 and legacy missing markers describe the same metric definitions.
        if version is not None and (version.endswith("v2") or index % 2):
            row["metrics"]["metric_schema_version"] = version
    assert all(
        signal.status == "clear" for signal in scan_campaign(rows, detector_ids=[detector]).signals
    )
    assert detect(detector, rows[0], cohort=rows).status == "clear"
