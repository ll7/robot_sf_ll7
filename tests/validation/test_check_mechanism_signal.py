"""Tests for the trace-pair mechanism-signal checker."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.validation import check_mechanism_signal

if TYPE_CHECKING:
    from pathlib import Path


def _trace(
    *,
    x: float = 0.0,
    command: float = 0.1,
    activation_count: int = 0,
    mechanism_score: float = 0.0,
    outcome: str = "timeout",
) -> dict[str, object]:
    """Return a tiny simulation-trace-like payload."""
    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": "fixture",
        "terminal_outcome": outcome,
        "algorithm_metadata": {
            "static_recenter": {
                "activation_count": activation_count,
                "mechanism_score": mechanism_score,
            }
        },
        "frames": [
            {
                "step": 0,
                "time_s": 0.0,
                "robot": {"position": [0.0, 0.0]},
                "pedestrians": [],
                "planner": {"selected_action": {"linear_velocity": 0.1}},
            },
            {
                "step": 1,
                "time_s": 0.1,
                "robot": {"position": [x, 0.0]},
                "pedestrians": [],
                "planner": {"selected_action": {"linear_velocity": command}},
            },
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON fixture payload."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_identical_pair_classifies_as_rendering_sanity() -> None:
    """Zero-delta pairs should not be routed as mechanism evidence."""
    result = check_mechanism_signal.classify_mechanism_signal(_trace(), _trace())

    signal = result["mechanism_signal"]
    assert signal["schema_version"] == "mechanism_signal_check.v1"
    assert signal["trajectory_delta_nonzero"] is False
    assert signal["command_delta_nonzero"] is False
    assert signal["mechanism_field_delta_nonzero"] is False
    assert signal["activation_delta_nonzero"] is False
    assert signal["outcome_delta_nonzero"] is False
    assert signal["classification"] == "rendering_sanity"


def test_trajectory_or_command_only_delta_is_qualitative_illustration() -> None:
    """Nonzero behavior without mechanism fields should remain qualitative-only."""
    baseline = _trace(x=0.0, command=0.1)
    intervention = _trace(x=0.2, command=0.2)

    signal = check_mechanism_signal.classify_mechanism_signal(
        baseline,
        intervention,
    )["mechanism_signal"]

    assert signal["trajectory_delta_nonzero"] is True
    assert signal["command_delta_nonzero"] is True
    assert signal["mechanism_field_delta_nonzero"] is False
    assert signal["activation_delta_nonzero"] is False
    assert signal["classification"] == "qualitative_illustration"


def test_activation_or_mechanism_delta_is_candidate() -> None:
    """Mechanism or activation field deltas should become mechanism candidates."""
    baseline = _trace(activation_count=0, mechanism_score=0.0)
    intervention = _trace(activation_count=2, mechanism_score=1.5)

    signal = check_mechanism_signal.classify_mechanism_signal(
        baseline,
        intervention,
    )["mechanism_signal"]

    assert signal["mechanism_field_delta_nonzero"] is True
    assert signal["activation_delta_nonzero"] is True
    assert signal["classification"] == "mechanism_difference_candidate"
    assert "does not establish planner superiority" in signal["claim_boundary"]


def test_empty_mechanism_section_presence_is_candidate() -> None:
    """Mechanism-section presence should count even without scalar leaves."""
    baseline = _trace()
    intervention = _trace()
    baseline["algorithm_metadata"] = {}
    intervention["algorithm_metadata"] = {"static_recenter": {}}

    signal = check_mechanism_signal.classify_mechanism_signal(
        baseline,
        intervention,
    )["mechanism_signal"]

    assert signal["mechanism_field_delta_nonzero"] is True
    assert signal["classification"] == "mechanism_difference_candidate"


def test_cli_writes_json_output(tmp_path: Path, capsys) -> None:
    """CLI should print and optionally persist the classification payload."""
    baseline = tmp_path / "baseline.json"
    intervention = tmp_path / "intervention.json"
    output_json = tmp_path / "signal.json"
    _write_json(baseline, _trace())
    _write_json(intervention, _trace(activation_count=1))

    exit_code = check_mechanism_signal.main(
        [
            "--baseline-trace",
            str(baseline),
            "--intervention-trace",
            str(intervention),
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["mechanism_signal"]["classification"] == "mechanism_difference_candidate"
