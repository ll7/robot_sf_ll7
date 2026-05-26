"""Tests for camera-ready benchmark CLI entrypoint."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.tools import run_camera_ready_benchmark

if TYPE_CHECKING:
    from pathlib import Path


def test_main_preflight_mode_emits_preflight_payload(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """CLI preflight mode should return only preflight/matrix artifact paths."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    sentinel_cfg = object()
    called: dict[str, bool] = {"preflight": False, "run": False}

    def _fake_load_campaign_config(path: Path):
        """Return the sentinel config only for the requested config path."""
        assert path == config_path
        return sentinel_cfg

    def _fake_prepare_campaign_preflight(cfg, **kwargs):
        """Return preflight artifact paths and record that preflight ran."""
        assert cfg is sentinel_cfg
        assert kwargs["campaign_id"] == "fixed-campaign"
        called["preflight"] = True
        return {
            "campaign_id": "cid",
            "campaign_root": tmp_path / "out" / "cid",
            "validate_config_path": tmp_path / "out" / "cid" / "preflight" / "validate_config.json",
            "preview_scenarios_path": tmp_path
            / "out"
            / "cid"
            / "preflight"
            / "preview_scenarios.json",
            "matrix_summary_json_path": tmp_path
            / "out"
            / "cid"
            / "reports"
            / "matrix_summary.json",
            "matrix_summary_csv_path": tmp_path / "out" / "cid" / "reports" / "matrix_summary.csv",
            "amv_coverage_json_path": tmp_path
            / "out"
            / "cid"
            / "reports"
            / "amv_coverage_summary.json",
            "amv_coverage_md_path": tmp_path
            / "out"
            / "cid"
            / "reports"
            / "amv_coverage_summary.md",
            "comparability_json_path": tmp_path
            / "out"
            / "cid"
            / "reports"
            / "comparability_matrix.json",
            "comparability_md_path": tmp_path
            / "out"
            / "cid"
            / "reports"
            / "comparability_matrix.md",
        }

    def _fake_run_campaign(*args, **kwargs):
        """Fail if run mode is invoked while testing preflight mode."""
        del args, kwargs
        called["run"] = True
        raise AssertionError("run_campaign should not be called in preflight mode")

    monkeypatch.setattr(
        run_camera_ready_benchmark, "load_campaign_config", _fake_load_campaign_config
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "prepare_campaign_preflight",
        _fake_prepare_campaign_preflight,
    )
    monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run_campaign)

    exit_code = run_camera_ready_benchmark.main(
        [
            "--config",
            str(config_path),
            "--mode",
            "preflight",
            "--campaign-id",
            "fixed-campaign",
        ],
    )
    assert exit_code == 0
    assert called["preflight"] is True
    assert called["run"] is False
    payload = json.loads(capsys.readouterr().out)
    assert set(payload.keys()) == {
        "campaign_id",
        "campaign_root",
        "validate_config_path",
        "preview_scenarios_path",
        "matrix_summary_json",
        "matrix_summary_csv",
        "amv_coverage_json",
        "amv_coverage_md",
        "comparability_json",
        "comparability_md",
    }


def test_main_run_mode_uses_run_campaign(tmp_path: Path, monkeypatch, capsys) -> None:
    """CLI run mode should call run_campaign and forward its payload."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    sentinel_cfg = object()
    called: dict[str, bool] = {"preflight": False, "run": False}

    def _fake_load_campaign_config(path: Path):
        """Return the sentinel config only for the requested run-mode path."""
        assert path == config_path
        return sentinel_cfg

    def _fake_prepare_campaign_preflight(*args, **kwargs):
        """Fail if preflight is invoked while testing run mode."""
        del args, kwargs
        called["preflight"] = True
        raise AssertionError("prepare_campaign_preflight should not be called in run mode")

    def _fake_run_campaign(cfg, **kwargs):
        """Return a minimal successful campaign payload for run mode."""
        assert cfg is sentinel_cfg
        assert isinstance(kwargs.get("invoked_command"), str)
        assert kwargs["campaign_id"] == "fixed-campaign"
        called["run"] = True
        return {
            "campaign_id": "cid",
            "campaign_root": str(tmp_path / "out" / "cid"),
            "benchmark_success": True,
            "status": "benchmark_success",
            "campaign_execution_status": "completed",
            "evidence_status": "valid",
            "row_status_summary": {
                "successful_evidence_rows": 1,
                "accepted_unavailable_rows": 0,
                "unexpected_failed_rows": 0,
                "fallback_or_degraded_rows": 0,
            },
            "status_reason": "all planner rows were benchmark-success",
            "successful_runs": 1,
            "accepted_unavailable_runs": 0,
            "unexpected_failed_runs": 0,
            "non_success_runs": 0,
            "total_runs": 1,
        }

    monkeypatch.setattr(
        run_camera_ready_benchmark, "load_campaign_config", _fake_load_campaign_config
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "prepare_campaign_preflight",
        _fake_prepare_campaign_preflight,
    )
    monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run_campaign)

    exit_code = run_camera_ready_benchmark.main(
        ["--config", str(config_path), "--campaign-id", "fixed-campaign"]
    )
    assert exit_code == 0
    assert called["run"] is True
    assert called["preflight"] is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["campaign_id"] == "cid"
    assert payload["campaign_execution_status"] == "completed"
    assert payload["evidence_status"] == "valid"


def test_main_run_mode_returns_non_zero_for_non_success_campaign(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Run mode must fail closed when campaign result is not benchmark-success."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    sentinel_cfg = object()

    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "load_campaign_config",
        lambda path: sentinel_cfg if path == config_path else None,
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "prepare_campaign_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepare_campaign_preflight should not be called in run mode")
        ),
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "run_campaign",
        lambda cfg, **kwargs: (
            {
                "campaign_id": "cid",
                "campaign_root": str(tmp_path / "out" / "cid"),
                "benchmark_success": False,
                "campaign_execution_status": "failed",
                "evidence_status": "invalid",
                "row_status_summary": {
                    "successful_evidence_rows": 0,
                    "accepted_unavailable_rows": 0,
                    "unexpected_failed_rows": 1,
                    "fallback_or_degraded_rows": 0,
                },
            }
            if cfg is sentinel_cfg and isinstance(kwargs.get("invoked_command"), str)
            else {}
        ),
    )

    exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark_success"] is False
    assert payload["campaign_execution_status"] == "failed"
    assert payload["evidence_status"] == "invalid"


def test_main_run_mode_returns_exit_code_3_for_accepted_unavailable_only_campaign(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Accepted-unavailable-only campaigns should stay non-success with exit code 3."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    sentinel_cfg = object()

    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "load_campaign_config",
        lambda path: sentinel_cfg if path == config_path else None,
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "prepare_campaign_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepare_campaign_preflight should not be called in run mode")
        ),
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "run_campaign",
        lambda cfg, **kwargs: (
            {
                "campaign_id": "cid",
                "campaign_root": str(tmp_path / "out" / "cid"),
                "benchmark_success": False,
                "status": "accepted_unavailable_only",
                "campaign_execution_status": "completed",
                "evidence_status": "partial",
                "row_status_summary": {
                    "successful_evidence_rows": 1,
                    "accepted_unavailable_rows": 1,
                    "unexpected_failed_rows": 0,
                    "fallback_or_degraded_rows": 1,
                },
                "status_reason": (
                    "campaign contains accepted unavailable/excluded rows and no unexpected failed rows"
                ),
                "successful_runs": 1,
                "accepted_unavailable_runs": 1,
                "unexpected_failed_runs": 0,
                "non_success_runs": 1,
                "total_runs": 2,
            }
            if cfg is sentinel_cfg and isinstance(kwargs.get("invoked_command"), str)
            else {}
        ),
    )

    exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])

    assert exit_code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "accepted_unavailable_only"
    assert payload["benchmark_success"] is False
    assert payload["campaign_execution_status"] == "completed"
    assert payload["evidence_status"] == "partial"
