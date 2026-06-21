"""Tests for the signalized crossing failure case pack builder script."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

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
    assert case["trace_row_range"] == [0, 1]
    assert case["signal_phase"] == "red"
    assert case["stop_line_geometry"] == [[1.0, 1.0], [1.0, -1.0]]
    assert case["robot_state"]["position"] == [0.5, 0.0]  # Closest approach/failure step is frame 1
    assert case["denominator_status"] == "eligible"
    assert case["stale_current_status"] == "current"
    assert case["allowed_claim_wording"] == ALLOWED_CLAIM_WORDING
    assert case["diagnostic_only"] is False
    assert case["figure_eligible"] is True


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
