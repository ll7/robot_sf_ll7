"""Tests for camera-ready benchmark CLI entrypoint."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from scripts.tools import orca_rvo2_preflight as orca_preflight_cli
from scripts.tools import run_camera_ready_benchmark

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@contextmanager
def _rvo2_missing() -> Generator[None, None, None]:
    """Context manager that temporarily removes rvo2 from sys.modules."""
    saved = sys.modules.get("rvo2")
    sys.modules["rvo2"] = None
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("rvo2", None)
        else:
            sys.modules["rvo2"] = saved


def _nop_preflight(cfg: object) -> None:
    """No-op replacement for check_orca_rvo2_preflight."""
    del cfg


def _write_config(path: Path, *, algo: str, key: str | None = None) -> None:
    """Write a minimal campaign config with a resolvable scenario matrix path."""
    (path.parent / "scenarios.yaml").write_text("scenarios: []\n", encoding="utf-8")
    path.write_text(
        "name: test\n"
        "scenario_matrix: scenarios.yaml\n"
        "planners:\n"
        f"  - key: {key or algo}\n"
        f"    algo: {algo}\n",
        encoding="utf-8",
    )


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
    monkeypatch.setattr(run_camera_ready_benchmark, "check_orca_rvo2_preflight", _nop_preflight)
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
        }

    monkeypatch.setattr(
        run_camera_ready_benchmark, "load_campaign_config", _fake_load_campaign_config
    )
    monkeypatch.setattr(run_camera_ready_benchmark, "check_orca_rvo2_preflight", _nop_preflight)
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
    monkeypatch.setattr(run_camera_ready_benchmark, "check_orca_rvo2_preflight", _nop_preflight)
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
            }
            if cfg is sentinel_cfg and isinstance(kwargs.get("invoked_command"), str)
            else {}
        ),
    )

    exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark_success"] is False


class TestRunModeOrcaPreflightIntegration:
    """Integration tests that the CLI runner calls the ORCA-rvo2 preflight guard."""

    def test_run_mode_aborts_when_orca_config_rvo2_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run mode aborts before run_campaign when ORCA config has rvo2 missing."""
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, algo="orca")
        run_called = False

        def _fake_run_campaign(*args, **kwargs):
            del args, kwargs
            nonlocal run_called
            run_called = True

        monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run_campaign)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", _fake_run_campaign
        )

        with _rvo2_missing():
            with pytest.raises(SystemExit) as exc_info:
                run_camera_ready_benchmark.main(["--config", str(config_path)])
            assert "rvo2" in str(exc_info.value).lower()
        assert not run_called

    def test_run_mode_passes_when_no_orca_in_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run mode proceeds to run_campaign when config has no ORCA planners."""
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, algo="ppo")
        run_called = False

        def _fake_run_campaign(cfg, **kwargs):
            del cfg, kwargs
            nonlocal run_called
            run_called = True
            return {"campaign_id": "cid", "campaign_root": str(tmp_path), "benchmark_success": True}

        monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run_campaign)
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "prepare_campaign_preflight",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])
        assert exit_code == 0
        assert run_called


class TestOrcaRvo2PreflightCli:
    """Tests for the standalone orca_rvo2_preflight CLI script."""

    def test_cli_exits_zero_for_non_orca_config(self, tmp_path: Path) -> None:
        """CLI returns exit code 0 when config has no ORCA planners."""
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, algo="ppo")
        exit_code = orca_preflight_cli.main(["--config", str(config_path)])
        assert exit_code == 0

    def test_cli_exits_nonzero_when_rvo2_missing(self, tmp_path: Path) -> None:
        """CLI returns exit code 1 when ORCA config has rvo2 missing."""
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, algo="orca")
        with _rvo2_missing():
            exit_code = orca_preflight_cli.main(["--config", str(config_path)])
        assert exit_code == 1

    def test_cli_passes_when_rvo2_available(self, tmp_path: Path) -> None:
        """CLI returns exit code 0 when ORCA config has rvo2 importable."""
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, algo="orca")
        with patch.dict(sys.modules, {"rvo2": True}):
            exit_code = orca_preflight_cli.main(["--config", str(config_path)])
        assert exit_code == 0

    def test_cli_errors_for_missing_config_file(self, tmp_path: Path) -> None:
        """CLI returns exit code 1 when config file does not exist."""
        config_path = tmp_path / "nonexistent.yaml"
        exit_code = orca_preflight_cli.main(["--config", str(config_path)])
        assert exit_code == 1
