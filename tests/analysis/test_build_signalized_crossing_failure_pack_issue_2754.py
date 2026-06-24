"""Tests for the signalized crossing failure case pack builder script."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.analysis.build_signalized_crossing_failure_pack_issue_2754 import (
    ALLOWED_CLAIM_WORDING,
    _portable_input_path,
    main,
)


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


def test_portable_input_path_is_repo_relative_outside_repo_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repo-local absolute paths stay repo-relative even when the process CWD is elsewhere."""
    repo_root = Path(__file__).resolve().parents[2]
    source_path = (
        repo_root / "scripts/analysis/build_signalized_crossing_failure_pack_issue_2754.py"
    )

    monkeypatch.chdir(tmp_path)

    assert (
        _portable_input_path(source_path)
        == "scripts/analysis/build_signalized_crossing_failure_pack_issue_2754.py"
    )


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


def test_top_level_fields_presence(tmp_path: Path) -> None:
    """Proves that top-level diagnostic_only and figure_eligible fields are added to the JSON."""
    trace_path, record_path = _write_failure_inputs(tmp_path)
    output_path = tmp_path / "result.json"

    # 1. Failures present, eligible run parameters
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
    result = json.loads(output_path.read_text())
    assert result["negative_control"] is False
    assert result["diagnostic_only"] is False
    assert result["figure_eligible"] is True

    # 2. Failures present, but ineligible (smoke tier)
    exit_code = main(
        [
            "--traces",
            str(trace_path),
            "--episodes-jsonl",
            str(record_path),
            "--output-json",
            str(output_path),
            "--evidence-tier",
            "smoke",
        ]
    )
    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["negative_control"] is False
    assert result["diagnostic_only"] is True
    assert result["figure_eligible"] is False

    # 3. Negative control, run-level eligible
    # Write a record that is a success (no failures)
    record = {
        "episode_id": "success_ep",
        "scenario_id": "success_scen",
        "metrics": {"collisions": 0.0, "comfort_exposure": 0.0, "near_misses": 0},
    }
    record_path.write_text(json.dumps(record) + "\n")
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
    result = json.loads(output_path.read_text())
    assert result["negative_control"] is True
    assert result["diagnostic_only"] is False
    assert result["figure_eligible"] is True

    # 4. Negative control, run-level ineligible
    exit_code = main(
        [
            "--traces",
            str(trace_path),
            "--episodes-jsonl",
            str(record_path),
            "--output-json",
            str(output_path),
            "--evidence-tier",
            "smoke",
        ]
    )
    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["negative_control"] is True
    assert result["diagnostic_only"] is True
    assert result["figure_eligible"] is False


@pytest.mark.parametrize("status", ["stale", "unknown"])
def test_artifact_status_stale_and_unknown(tmp_path: Path, status: str) -> None:
    """Explicitly verify that artifact_status=stale and artifact_status=unknown fail figure eligibility."""
    trace_path, record_path = _write_failure_inputs(tmp_path)
    output_path = tmp_path / "result.json"

    args = _eligible_runtime_args()
    # Replace default allowed/current status with custom ones
    exit_code = main(
        [
            "--traces",
            str(trace_path),
            "--episodes-jsonl",
            str(record_path),
            "--output-json",
            str(output_path),
            *args,
            "--artifact-status",
            status,
        ]
    )
    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["diagnostic_only"] is True
    assert result["figure_eligible"] is False
    case = result["cases"][0]
    assert case["diagnostic_only"] is True
    assert case["figure_eligible"] is False
    assert f"artifact_status={status}" in case["figure_ineligibility_reasons"]


def test_scanner_finds_disagreements(tmp_path: Path) -> None:
    """Verify that run_evidence_scan detects prose and machine-readable eligibility disagreements."""
    from scripts.analysis.build_signalized_crossing_failure_pack_issue_2754 import run_evidence_scan

    # Create dummy evidence directory structure
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    # 1. Clean subdir: prose contains 'smoke', JSON correctly has figure_eligible: False
    clean_dir = evidence_dir / "clean_smoke"
    clean_dir.mkdir()
    (clean_dir / "README.md").write_text("This records smoke test results.")
    (clean_dir / "summary.json").write_text(
        json.dumps({"figure_eligible": False, "diagnostic_only": True})
    )

    # 2. Conflicting subdir: prose contains 'synthetic', JSON has figure_eligible: True
    conflict_dir = evidence_dir / "conflict_synthetic"
    conflict_dir.mkdir()
    (conflict_dir / "README.md").write_text("This records synthetic traces.")
    (conflict_dir / "summary.json").write_text(
        json.dumps({"cases": [{"figure_eligible": True, "diagnostic_only": False}]})
    )

    # 3. Another conflicting subdir: prose contains 'fixture', JSON has diagnostic_only: False
    conflict_dir2 = evidence_dir / "conflict_fixture"
    conflict_dir2.mkdir()
    (conflict_dir2 / "README.md").write_text("This contains fixture only records.")
    (conflict_dir2 / "summary.json").write_text(json.dumps({"diagnostic_only": False}))

    # Run scan
    exit_code = run_evidence_scan(evidence_dir)
    assert exit_code == 1

    # Check output/clean state when conflict dirs are removed
    import shutil

    shutil.rmtree(conflict_dir)
    shutil.rmtree(conflict_dir2)

    exit_code_clean = run_evidence_scan(evidence_dir)
    assert exit_code_clean == 0


def test_scanner_finds_nested_disagreements(tmp_path: Path) -> None:
    """Evidence scans recurse into nested artifact directories."""
    from scripts.analysis.build_signalized_crossing_failure_pack_issue_2754 import run_evidence_scan

    evidence_dir = tmp_path / "evidence"
    nested_dir = evidence_dir / "parent" / "nested_smoke"
    nested_dir.mkdir(parents=True)
    (nested_dir / "README.md").write_text("Nested smoke evidence.")
    (nested_dir / "summary.json").write_text(json.dumps({"figure_eligible": True}))

    assert run_evidence_scan(evidence_dir) == 1


def test_signal_red_phase_violation_can_define_failure_case(tmp_path: Path) -> None:
    """Signal-specific red violations are an explicit failure-pack predicate."""
    trace_data = _make_dummy_trace("signal_fail_ep", "signal_fail_scen")
    trace_path = tmp_path / "signal_trace.json"
    trace_path.write_text(json.dumps(trace_data))
    record = {
        "episode_id": "signal_fail_ep",
        "scenario_id": "signal_fail_scen",
        "seed": 123,
        "metrics": {
            "collisions": 0.0,
            "comfort_exposure": 0.0,
            "near_misses": 0,
            "signal_red_phase_violations": 1,
            "signal_stop_line_crossings_under_red": 1,
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
            *_eligible_runtime_args(),
        ]
    )

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["negative_control"] is False
    assert result["status"] == "failures_present"
    assert len(result["cases"]) == 1
    case = result["cases"][0]
    assert case["signal_failure_predicates"] == [
        "signal_red_phase_violations",
        "signal_stop_line_crossings_under_red",
    ]
    assert case["metric_row"]["signal_red_phase_violations"] == 1
    assert case["metric_row"]["collisions"] == 0.0
    assert case["figure_eligible"] is True


def test_signal_failure_predicate_requires_positive_signal_metrics(tmp_path: Path) -> None:
    """Zero or malformed signal-specific metrics do not create false positive cases."""
    trace_data = _make_dummy_trace("signal_nonfail_ep", "signal_nonfail_scen")
    trace_path = tmp_path / "signal_trace.json"
    trace_path.write_text(json.dumps(trace_data))
    record = {
        "episode_id": "signal_nonfail_ep",
        "scenario_id": "signal_nonfail_scen",
        "seed": 123,
        "metrics": {
            "collisions": 0.0,
            "comfort_exposure": 0.0,
            "near_misses": 0,
            "signal_red_phase_violations": "not-a-number",
            "signal_stop_line_crossings_under_red": 0,
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
            *_eligible_runtime_args(),
        ]
    )

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["negative_control"] is True
    assert result["status"] == "insufficiently_adversarial"
    assert result["cases"] == []
