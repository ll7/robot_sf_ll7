"""Tests for the live forecast replay gate validation CLI (issue #2941)."""

from __future__ import annotations

import json

import pytest

from robot_sf.benchmark.live_forecast_replay_gate import (
    FORECAST_VARIANTS,
    RUN_CLASSIFICATION_NATIVE,
    SMOKE_FORECAST_VARIANTS,
)
from scripts.validation import validate_live_forecast_replay_gate


def test_cli_emits_json_error_for_missing_trace(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing trace input should fail closed with machine-readable JSON."""

    missing_trace = tmp_path / "missing_trace.json"

    exit_code = validate_live_forecast_replay_gate.main(["--trace", str(missing_trace)])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "No such file" in payload["error"] or "not found" in payload["error"]


def test_cli_emits_json_error_when_no_requested_horizons_are_feasible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unavailable horizons should fail closed through the CLI, not traceback."""

    exit_code = validate_live_forecast_replay_gate.main(
        [
            "--trace",
            "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
            "dense_pedestrian_stress_episode_0000.json",
            "--horizon-s",
            "99",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "error",
        "error": "no requested forecast horizons fit within the trace duration",
    }


def test_cli_default_runs_smoke_variants(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI default should run only the none+cv smoke variants."""

    exit_code = validate_live_forecast_replay_gate.main(
        [
            "--trace",
            "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
            "dense_pedestrian_stress_episode_0000.json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["variant_results"]) == set(SMOKE_FORECAST_VARIANTS)
    assert payload["classification"] == RUN_CLASSIFICATION_NATIVE
    assert payload["full_matrix_expansion_recommended"] is True


def test_cli_full_matrix_runs_all_variants(capsys: pytest.CaptureFixture[str]) -> None:
    """The --full-matrix flag should evaluate all five forecast variants."""

    exit_code = validate_live_forecast_replay_gate.main(
        [
            "--trace",
            "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
            "dense_pedestrian_stress_episode_0000.json",
            "--full-matrix",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["variant_results"]) == set(FORECAST_VARIANTS)
    assert payload["classification"] == RUN_CLASSIFICATION_NATIVE
