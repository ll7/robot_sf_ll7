"""Tests for the signalized crossing failure case pack builder script."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.analysis.build_signalized_crossing_failure_pack_issue_2754 import (
    ALLOWED_CLAIM_WORDING,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_dummy_trace(episode_id: str, scenario_id: str) -> dict[str, Any]:
    """Build a minimal valid simulation_trace_export.v1 payload."""
    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": f"{episode_id}_trace",
        "source": {
            "scenario_id": scenario_id,
            "seed": 123,
            "planner_id": "dummy_planner",
            "episode_id": episode_id,
            "generated_by": "test",
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": "world",
        "units": {
            "position": "m",
            "heading": "rad",
            "time": "s",
            "velocity": "m/s",
        },
        "frames": [
            {
                "step": 0,
                "time_s": 0.0,
                "robot": {"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                "pedestrians": [
                    {
                        "id": "1",
                        "position": [2.0, 0.0],
                        "velocity": [0.0, 0.0],
                    }
                ],
                "planner": {
                    "selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
                    "event": "start",
                },
            },
            {
                "step": 1,
                "time_s": 0.1,
                "robot": {"position": [0.5, 0.0], "heading": 0.0, "velocity": [5.0, 0.0]},
                "pedestrians": [
                    {
                        "id": "1",
                        "position": [0.6, 0.0],
                        "velocity": [0.0, 0.0],
                    }
                ],
                "planner": {
                    "selected_action": {"linear_velocity": 5.0, "angular_velocity": 0.0},
                    "event": "step1",
                },
            },
        ],
    }


def _write_failure_inputs(
    tmp_path: Path,
    *,
    episode_id: str = "fail_ep_0",
    scenario_id: str = "fail_scen_0",
    denominator: int = 1,
    evidence_state: str = "planner_observable",
    exclusion_reason: str = "",
) -> tuple[Path, Path]:
    """Write a paired failing trace and metric row for CLI tests."""
    trace_path = tmp_path / f"{episode_id}_trace.json"
    trace_path.write_text(json.dumps(_make_dummy_trace(episode_id, scenario_id)))
    record = {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": 123,
        "metrics": {
            "collisions": 1.0,
            "comfort_exposure": 0.0,
            "near_misses": 0,
            "signal_metrics_denominator": denominator,
            "signal_metrics_evidence": {
                "state": evidence_state,
                "exclusion_reason": exclusion_reason,
            },
        },
        "scenario_params": {
            "metadata": {
                "signal_state": {
                    "timeline": [{"state": "red", "duration": 5.0}],
                    "stop_line": [[1.0, 1.0], [1.0, -1.0]],
                }
            }
        },
    }
    record_path = tmp_path / f"{episode_id}_episodes.jsonl"
    record_path.write_text(json.dumps(record) + "\n")
    return trace_path, record_path


def _eligible_runtime_args() -> list[str]:
    """Return provenance flags for a current, claimable runtime-backed row."""
    return [
        "--trace-source-kind",
        "live_execution",
        "--metric-source-kind",
        "live_execution",
        "--execution-performed",
        "--evidence-tier",
        "benchmark",
        "--execution-mode",
        "native",
        "--claim-matrix-status",
        "allowed",
    ]


def test_real_failure_case_output(tmp_path: Path) -> None:
    """Proves that a real failure case triggers non-negative-control output and contains all expected fields."""
    trace_data = _make_dummy_trace("fail_ep_0", "fail_scen_0")
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace_data))

    record = {
        "episode_id": "fail_ep_0",
        "scenario_id": "fail_scen_0",
        "seed": 123,
        "metrics": {
            "collisions": 1.0,
            "comfort_exposure": 0.0,
            "near_misses": 0,
            "signal_metrics_denominator": 1,
            "signal_metrics_evidence": {
                "state": "planner_observable",
                "exclusion_reason": "",
            },
        },
        "scenario_params": {
            "metadata": {
                "signal_state": {
                    "timeline": [{"state": "red", "duration": 5.0}],
                    "stop_line": [[1.0, 1.0], [1.0, -1.0]],
                }
            }
        },
    }
    record_path = tmp_path / "episodes.jsonl"
    record_path.write_text(json.dumps(record) + "\n")

    output_path = tmp_path / "result.json"

    # Call via main
    exit_code = main(
        [
            "--traces",
            str(trace_path),
            "--episodes-jsonl",
            str(record_path),
            "--output-json",
            str(output_path),
            *_eligible_runtime_args(),
        ]
    )
    assert exit_code == 0
    assert output_path.exists()

    result = json.loads(output_path.read_text())
    assert result["schema_version"] == "signalized_crossing_failure_pack.v1"
    assert result["negative_control"] is False
    assert result["status"] == "failures_present"
    assert len(result["cases"]) == 1

    case = result["cases"][0]
    assert case["episode_id"] == "fail_ep_0"
    assert case["scenario_id"] == "fail_scen_0"
    assert case["trace_path"] == "trace.json"
    assert case["episodes_jsonl_path"] == "episodes.jsonl"
    assert case["metric_row_line_number"] == 1
    assert case["metric_row_claim_boundary"] is None
    assert case["trace_row_range"] == [0, 1]
    assert case["signal_phase"] == "red"
    assert case["stop_line_geometry"] == [[1.0, 1.0], [1.0, -1.0]]
    assert case["robot_state"]["position"] == [0.5, 0.0]  # Closest approach/failure step is frame 1
    assert case["denominator_status"] == "eligible"
    assert case["stale_current_status"] == "current"
    assert case["artifact_status"] == "current"
    assert case["trace_source_kind"] == "live_execution"
    assert case["metric_source_kind"] == "live_execution"
    assert case["execution_performed"] is True
    assert case["evidence_tier"] == "benchmark"
    assert case["execution_mode"] == "native"
    assert case["fallback_or_degraded"] is False
    assert case["claim_matrix_status"] == "allowed"
    assert case["figure_ineligibility_reasons"] == []
    assert case["allowed_claim_wording"] == ALLOWED_CLAIM_WORDING
    assert case["diagnostic_only"] is False
    assert case["figure_eligible"] is True


def test_metric_row_input_provenance_records_matched_jsonl_line(tmp_path: Path) -> None:
    """The emitted case points to the matched metric row, not just the JSONL file."""
    trace_data = _make_dummy_trace("fail_ep_0", "fail_scen_0")
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace_data))

    unmatched_record = {
        "episode_id": "other_ep_0",
        "scenario_id": "other_scen_0",
        "seed": 456,
        "metrics": {
            "collisions": 1.0,
            "comfort_exposure": 0.0,
            "near_misses": 0,
            "signal_metrics_denominator": 1,
            "signal_metrics_evidence": {
                "state": "planner_observable",
                "exclusion_reason": "",
            },
        },
    }
    matched_record = {
        "episode_id": "fail_ep_0",
        "scenario_id": "fail_scen_0",
        "seed": 123,
        "metrics": {
            "collisions": 1.0,
            "comfort_exposure": 0.0,
            "near_misses": 0,
            "signal_metrics_denominator": 1,
            "signal_metrics_evidence": {
                "state": "planner_observable",
                "exclusion_reason": "",
            },
        },
        "scenario_params": {
            "metadata": {
                "signal_state": {
                    "timeline": [{"state": "red", "duration": 5.0}],
                    "stop_line": [[1.0, 1.0], [1.0, -1.0]],
                }
            }
        },
        "claim_boundary": "unit-test metric row boundary",
    }
    record_path = tmp_path / "episodes.jsonl"
    record_path.write_text(
        json.dumps(unmatched_record) + "\n" + json.dumps(matched_record) + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "result.json"

    exit_code = main(
        [
            "--traces",
            str(trace_path),
            "--episodes-jsonl",
            str(record_path),
            "--output-json",
            str(output_path),
            *_eligible_runtime_args(),
        ]
    )
    assert exit_code == 0

    result = json.loads(output_path.read_text(encoding="utf-8"))
    case = result["cases"][0]
    assert case["trace_path"] == "trace.json"
    assert case["episodes_jsonl_path"] == "episodes.jsonl"
    assert case["metric_row_line_number"] == 2
    assert case["metric_row_claim_boundary"] == "unit-test metric row boundary"
    assert not case["trace_path"].startswith("/")
    assert not case["episodes_jsonl_path"].startswith("/")


def test_invalid_jsonl_error_reports_input_path_and_line(tmp_path: Path) -> None:
    """Malformed metric rows fail with the JSONL source path and physical line number."""
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_make_dummy_trace("fail_ep_0", "fail_scen_0")))
    record_path = tmp_path / "episodes.jsonl"
    record_path.write_text(json.dumps({"episode_id": "other_ep_0"}) + "\n" + "{oops\n")
    output_path = tmp_path / "result.json"

    with pytest.raises(ValueError) as exc_info:
        main(
            [
                "--traces",
                str(trace_path),
                "--episodes-jsonl",
                str(record_path),
                "--output-json",
                str(output_path),
            ]
        )

    message = str(exc_info.value)
    assert "Invalid JSONL in episodes.jsonl at line 2" in message
    assert "Expecting property name enclosed in double quotes" in message


def test_non_object_jsonl_error_reports_input_path_and_line(tmp_path: Path) -> None:
    """Metric JSONL rows must be objects before provenance fields are attached."""
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_make_dummy_trace("fail_ep_0", "fail_scen_0")))
    record_path = tmp_path / "episodes.jsonl"
    record_path.write_text(json.dumps({"episode_id": "other_ep_0"}) + "\n" + "[]\n")
    output_path = tmp_path / "result.json"

    with pytest.raises(ValueError) as exc_info:
        main(
            [
                "--traces",
                str(trace_path),
                "--episodes-jsonl",
                str(record_path),
                "--output-json",
                str(output_path),
            ]
        )

    assert "Invalid JSONL in episodes.jsonl at line 2: expected object" in str(exc_info.value)


def test_missing_provenance_defaults_to_diagnostic_only(tmp_path: Path) -> None:
    """Missing provenance fails closed even when denominator evidence is planner-observable."""
    trace_path, record_path = _write_failure_inputs(tmp_path)
    output_path = tmp_path / "result.json"

    exit_code = main(
        [
            "--traces",
            str(trace_path),
            "--episodes-jsonl",
            str(record_path),
            "--output-json",
            str(output_path),
        ]
    )
    assert exit_code == 0

    result = json.loads(output_path.read_text())
    case = result["cases"][0]
    assert case["denominator_status"] == "eligible"
    assert case["diagnostic_only"] is True
    assert case["figure_eligible"] is False
    assert "trace_source_kind=unknown" in case["figure_ineligibility_reasons"]
    assert "metric_source_kind=unknown" in case["figure_ineligibility_reasons"]
    assert "execution_performed=false" in case["figure_ineligibility_reasons"]
    assert "diagnostic-only" in case["allowed_claim_wording"]


@pytest.mark.parametrize(
    ("extra_args", "expected_reason", "evidence_state", "exclusion_reason"),
    [
        (["--trace-source-kind", "fixture"], "trace_source_kind=fixture", "planner_observable", ""),
        (
            ["--metric-source-kind", "synthetic"],
            "metric_source_kind=synthetic",
            "planner_observable",
            "",
        ),
        (["--evidence-tier", "smoke"], "evidence_tier=smoke", "planner_observable", ""),
        (["--artifact-status", "stale"], "artifact_status=stale", "planner_observable", ""),
        (["--artifact-status", "unknown"], "artifact_status=unknown", "planner_observable", ""),
        (["--execution-mode", "fallback"], "execution_mode=fallback", "planner_observable", ""),
        (
            ["--execution-mode", "degraded", "--fallback-or-degraded"],
            "execution_mode=degraded",
            "planner_observable",
            "",
        ),
        (
            [],
            "signal_metrics_evidence.state=proxy_diagnostic",
            "proxy_diagnostic",
            "signal_state_not_benchmark_evidence",
        ),
        (
            [],
            "signal_metrics_evidence.state=unavailable",
            "unavailable",
            "signal_state_missing",
        ),
    ],
)
def test_provenance_or_signal_caveats_fail_closed(
    tmp_path: Path,
    extra_args: list[str],
    expected_reason: str,
    evidence_state: str,
    exclusion_reason: str,
) -> None:
    """Fixture, smoke, stale, degraded, proxy, and unavailable inputs are never figure-eligible."""
    trace_path, record_path = _write_failure_inputs(
        tmp_path,
        evidence_state=evidence_state,
        exclusion_reason=exclusion_reason,
    )
    output_path = tmp_path / "result.json"
    args = _eligible_runtime_args()
    for flag in (
        "--trace-source-kind",
        "--metric-source-kind",
        "--evidence-tier",
        "--execution-mode",
    ):
        if flag in extra_args:
            flag_index = args.index(flag)
            del args[flag_index : flag_index + 2]

    exit_code = main(
        [
            "--traces",
            str(trace_path),
            "--episodes-jsonl",
            str(record_path),
            "--output-json",
            str(output_path),
            *args,
            *extra_args,
        ]
    )
    assert exit_code == 0

    result = json.loads(output_path.read_text())
    case = result["cases"][0]
    assert case["diagnostic_only"] is True
    assert case["figure_eligible"] is False
    assert expected_reason in case["figure_ineligibility_reasons"]
    assert "diagnostic-only" in case["allowed_claim_wording"]


def test_negative_control_output(tmp_path: Path) -> None:
    """Proves that if no failures exist, negative-control output is generated."""
    trace_data = _make_dummy_trace("success_ep_0", "success_scen_0")
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace_data))

    record = {
        "episode_id": "success_ep_0",
        "scenario_id": "success_scen_0",
        "seed": 123,
        "metrics": {
            "collisions": 0.0,
            "comfort_exposure": 0.0,
            "near_misses": 0,
            "signal_metrics_denominator": 1,
            "signal_metrics_evidence": {
                "state": "planner_observable",
                "exclusion_reason": "",
            },
        },
        "scenario_params": {
            "metadata": {
                "signal_state": {
                    "timeline": [{"state": "red", "duration": 5.0}],
                    "stop_line": [[1.0, 1.0], [1.0, -1.0]],
                }
            }
        },
    }
    record_path = tmp_path / "episodes.jsonl"
    record_path.write_text(json.dumps(record) + "\n")

    output_path = tmp_path / "result.json"

    exit_code = main(
        [
            "--traces",
            str(trace_path),
            "--episodes-jsonl",
            str(record_path),
            "--output-json",
            str(output_path),
        ]
    )
    assert exit_code == 0

    result = json.loads(output_path.read_text())
    assert result["negative_control"] is True
    assert result["status"] == "insufficiently_adversarial"
    assert len(result["cases"]) == 0


def test_proxy_unavailable_rows_downgraded(tmp_path: Path) -> None:
    """Proves that proxy/unavailable failure rows are downgraded to diagnostic-only/figure-ineligible."""
    trace_data = _make_dummy_trace("proxy_ep_0", "proxy_scen_0")
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace_data))

    record = {
        "episode_id": "proxy_ep_0",
        "scenario_id": "proxy_scen_0",
        "seed": 123,
        "metrics": {
            "collisions": 1.0,
            "comfort_exposure": 0.0,
            "near_misses": 0,
            "signal_metrics_denominator": 0,
            "signal_metrics_evidence": {
                "state": "proxy_diagnostic",
                "exclusion_reason": "signal_state_not_benchmark_evidence",
            },
        },
        "scenario_params": {
            "metadata": {
                "signal_state": {
                    "timeline": [{"state": "red", "duration": 5.0}],
                    "stop_line": [[1.0, 1.0], [1.0, -1.0]],
                }
            }
        },
    }
    record_path = tmp_path / "episodes.jsonl"
    record_path.write_text(json.dumps(record) + "\n")

    output_path = tmp_path / "result.json"

    exit_code = main(
        [
            "--traces",
            str(trace_path),
            "--episodes-jsonl",
            str(record_path),
            "--output-json",
            str(output_path),
        ]
    )
    assert exit_code == 0

    result = json.loads(output_path.read_text())
    assert result["negative_control"] is False
    assert len(result["cases"]) == 1

    case = result["cases"][0]
    assert case["denominator_status"] == "excluded"
    assert case["diagnostic_only"] is True
    assert case["figure_eligible"] is False
    assert "diagnostic-only" in case["allowed_claim_wording"]
    assert "ineligible" in case["allowed_claim_wording"]
