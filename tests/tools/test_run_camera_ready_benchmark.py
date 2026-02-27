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
        assert path == config_path
        return sentinel_cfg

    def _fake_prepare_campaign_preflight(cfg, **kwargs):
        del kwargs
        assert cfg is sentinel_cfg
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
        }

    def _fake_run_campaign(*args, **kwargs):
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
        ["--config", str(config_path), "--mode", "preflight"],
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
    }


def test_main_run_mode_uses_run_campaign(tmp_path: Path, monkeypatch, capsys) -> None:
    """CLI run mode should call run_campaign and forward its payload."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    sentinel_cfg = object()
    called: dict[str, bool] = {"preflight": False, "run": False}

    def _fake_load_campaign_config(path: Path):
        assert path == config_path
        return sentinel_cfg

    def _fake_prepare_campaign_preflight(*args, **kwargs):
        del args, kwargs
        called["preflight"] = True
        raise AssertionError("prepare_campaign_preflight should not be called in run mode")

    def _fake_run_campaign(cfg, **kwargs):
        assert cfg is sentinel_cfg
        assert isinstance(kwargs.get("invoked_command"), str)
        called["run"] = True
        return {"campaign_id": "cid", "campaign_root": str(tmp_path / "out" / "cid")}

    monkeypatch.setattr(
        run_camera_ready_benchmark, "load_campaign_config", _fake_load_campaign_config
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "prepare_campaign_preflight",
        _fake_prepare_campaign_preflight,
    )
    monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run_campaign)

    exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])
    assert exit_code == 0
    assert called["run"] is True
    assert called["preflight"] is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["campaign_id"] == "cid"
