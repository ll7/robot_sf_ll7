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


class TestRunModeOrcaPreflightIntegration:
    """Integration tests that the CLI runner calls the ORCA-rvo2 preflight guard."""

    def test_run_mode_emits_fail_closed_payload_when_orca_config_rvo2_missing(
        self, tmp_path: Path, capsys
    ) -> None:
        """Run mode should translate typed ORCA preflight failures into structured JSON."""
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, algo="orca")

        with _rvo2_missing():
            exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])

        assert exit_code == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["mode"] == "run"
        assert payload["status"] == "orca_preflight_failed"
        assert payload["benchmark_success"] is False
        assert payload["campaign_execution_status"] == "failed"
        assert payload["evidence_status"] == "blocked"
        assert payload["exit_code"] == 2
        assert "rvo2" in payload["status_reason"].lower()

    def test_preflight_mode_emits_fail_closed_payload_when_orca_config_rvo2_missing(
        self, tmp_path: Path, capsys
    ) -> None:
        """Preflight mode should also stay fail-closed with actionable ORCA guidance."""
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, algo="orca")

        with _rvo2_missing():
            exit_code = run_camera_ready_benchmark.main(
                ["--config", str(config_path), "--mode", "preflight"]
            )

        assert exit_code == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["mode"] == "preflight"
        assert payload["status"] == "orca_preflight_failed"
        assert payload["benchmark_success"] is False
        assert payload["campaign_execution_status"] == "failed"
        assert payload["evidence_status"] == "blocked"
        assert payload["exit_code"] == 2
        assert "uv sync --extra orca" in payload["status_reason"]

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
            return {
                "campaign_id": "cid",
                "campaign_root": str(tmp_path),
                "benchmark_success": True,
                "successful_runs": 1,
                "non_success_runs": 0,
                "total_runs": 1,
            }

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

    def test_cli_errors_for_directory_config_path(self, tmp_path: Path) -> None:
        """CLI returns exit code 1 when config path is a directory."""
        exit_code = orca_preflight_cli.main(["--config", str(tmp_path)])
        assert exit_code == 1

    def test_cli_logs_clean_error_for_invalid_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI returns exit code 1 instead of a traceback for expected load failures."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: true\n", encoding="utf-8")

        def _raise_value_error(config_path: Path) -> None:
            raise ValueError(f"invalid config: {config_path}")

        monkeypatch.setattr(
            orca_preflight_cli,
            "check_orca_rvo2_preflight_from_config",
            _raise_value_error,
        )

        exit_code = orca_preflight_cli.main(["--config", str(config_path)])

        assert exit_code == 1


class TestPostCampaignStageStatusEnvelope:
    """Issue #5244 production-boundary regression for the camera-ready launcher.

    The canonical Python launcher must emit the
    ``robot-sf-post-campaign-stage-status.v1`` envelope at the real dispatch
    boundary so downstream schedulers/ledgers can separate a completed campaign
    from a failed reporting stage. The campaign exit code must be preserved.
    """

    def test_launcher_emits_stage_status_envelope_for_completed_campaign(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """A completed campaign must write the envelope and keep exit 0."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")
        sentinel_cfg = object()
        summary_path = tmp_path / "cid" / "reports" / "campaign_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"soft_contract_warning": True, "warnings": ["SNQI warn"]}),
            encoding="utf-8",
        )
        campaign_root = tmp_path / "cid"

        def _fake_load_campaign_config(path: Path):
            assert path == config_path
            return sentinel_cfg

        def _fake_run_campaign(cfg, **kwargs):
            assert cfg is sentinel_cfg
            return {
                "campaign_id": "cid",
                "campaign_root": str(campaign_root),
                "summary_json": str(summary_path),
                "benchmark_success": True,
                "status": "benchmark_success",
                "campaign_execution_status": "completed",
                "evidence_status": "valid",
                "soft_contract_warning": True,
                "row_status_summary": {
                    "successful_evidence_rows": 1,
                    "accepted_unavailable_rows": 0,
                    "unexpected_failed_rows": 0,
                    "fallback_or_degraded_rows": 0,
                },
                "successful_runs": 1,
                "accepted_unavailable_runs": 0,
                "unexpected_failed_runs": 0,
                "non_success_runs": 0,
                "total_runs": 1,
            }

        monkeypatch.setattr(
            run_camera_ready_benchmark, "load_campaign_config", _fake_load_campaign_config
        )
        monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run_campaign)
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "prepare_campaign_preflight",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])

        assert exit_code == 0
        envelope = campaign_root / "reports" / "post_campaign_stage_status.json"
        assert envelope.is_file()
        payload = json.loads(envelope.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "robot-sf-post-campaign-stage-status.v1"
        assert payload["campaign"]["exit_code"] == 0
        assert payload["campaign"]["status"] == "completed"
        assert payload["campaign"]["soft_contract_warning"] is True
        assert payload["job_exit_code"] == 0
        assert payload["post_campaign_stage"]["status"] == "completed"
        assert payload["post_campaign_stage"]["exit_code"] == 0
        assert payload["post_campaign_stage"]["name"] == "camera_ready_campaign"

    def test_launcher_preserves_nonzero_exit_when_campaign_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-success campaign must still exit nonzero and record the lane."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")
        sentinel_cfg = object()
        summary_path = tmp_path / "cid" / "reports" / "campaign_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("{}", encoding="utf-8")
        campaign_root = tmp_path / "cid"

        def _fake_load_campaign_config(path: Path):
            assert path == config_path
            return sentinel_cfg

        def _fake_run_campaign(cfg, **kwargs):
            assert cfg is sentinel_cfg
            return {
                "campaign_id": "cid",
                "campaign_root": str(campaign_root),
                "summary_json": str(summary_path),
                "benchmark_success": False,
                "status": "failed",
                "campaign_execution_status": "failed",
                "evidence_status": "invalid",
                "row_status_summary": {
                    "successful_evidence_rows": 0,
                    "accepted_unavailable_rows": 0,
                    "unexpected_failed_rows": 1,
                    "fallback_or_degraded_rows": 0,
                },
            }

        monkeypatch.setattr(
            run_camera_ready_benchmark, "load_campaign_config", _fake_load_campaign_config
        )
        monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run_campaign)
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "prepare_campaign_preflight",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])

        assert exit_code == 2
        payload = json.loads(
            (campaign_root / "reports" / "post_campaign_stage_status.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["campaign"]["exit_code"] == 2
        assert payload["campaign"]["status"] == "failed"
        assert payload["job_exit_code"] == 2


class TestLauncherToFinalizerHandoff:
    """Issue #5244 end-to-end regression at the production dispatch boundary.

    The real launcher ``main()`` writes the
    ``robot-sf-post-campaign-stage-status.v1`` envelope to disk; the real finalizer
    ``main()`` must then consume that exact on-disk artifact via
    ``--post-campaign-stage-status`` and preserve the campaign exit lane. This
    exercises the full serialization handoff (not a synthetic in-memory payload)
    and proves a completed ``enforcement=warn`` campaign cannot be remapped to a
    nonzero scheduler job exit by a downstream reporting-stage failure.
    """

    def _run_launcher(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        soft_warning: bool,
        benchmark_success: bool,
    ) -> tuple[int, Path]:
        """Drive the real launcher and return (exit_code, campaign_root)."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")
        sentinel_cfg = object()
        campaign_root = tmp_path / "cid"
        summary_path = campaign_root / "reports" / "campaign_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"soft_contract_warning": soft_warning, "warnings": ["SNQI warn"]})
            if soft_warning
            else "{}",
            encoding="utf-8",
        )

        def _fake_load_campaign_config(path: Path):
            assert path == config_path
            return sentinel_cfg

        def _fake_run_campaign(cfg, **kwargs):
            assert cfg is sentinel_cfg
            return {
                "campaign_id": "cid",
                "campaign_root": str(campaign_root),
                "summary_json": str(summary_path),
                "benchmark_success": benchmark_success,
                "status": "benchmark_success" if benchmark_success else "failed",
                "campaign_execution_status": "completed" if benchmark_success else "failed",
                "evidence_status": "valid" if benchmark_success else "invalid",
                "soft_contract_warning": soft_warning,
                "row_status_summary": {
                    "successful_evidence_rows": 1 if benchmark_success else 0,
                    "accepted_unavailable_rows": 0,
                    "unexpected_failed_rows": 0 if benchmark_success else 1,
                    "fallback_or_degraded_rows": 0,
                },
            }

        monkeypatch.setattr(
            run_camera_ready_benchmark, "load_campaign_config", _fake_load_campaign_config
        )
        monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run_campaign)
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "prepare_campaign_preflight",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        exit_code = run_camera_ready_benchmark.main(["--config", str(config_path)])
        return exit_code, campaign_root

    def test_completed_warn_campaign_keeps_exit_zero_through_finalizer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Launcher exit 0 + envelope must finalize as success with job_exit_code 0."""
        from scripts.tools import slurm_job_finalize

        launcher_exit, campaign_root = self._run_launcher(
            tmp_path, monkeypatch, soft_warning=True, benchmark_success=True
        )
        assert launcher_exit == 0

        envelope = campaign_root / "reports" / "post_campaign_stage_status.json"
        assert envelope.is_file()

        # The finalizer consumes the real on-disk artifact the launcher wrote.
        finalize_output = tmp_path / "finalization.json"
        finalizer_exit = slurm_job_finalize.main(
            [
                "--repo-root",
                str(tmp_path),
                "--issue",
                "5244",
                "--job-id",
                "13274",
                "--job-state",
                "COMPLETED",
                "--expected-artifact",
                str(campaign_root / "reports" / "campaign_summary.json"),
                "--post-campaign-stage-status",
                str(envelope),
                "--output",
                str(finalize_output),
            ]
        )

        assert finalizer_exit == 0
        report = json.loads(finalize_output.read_text(encoding="utf-8"))
        assert report["classification"] == "success"
        lanes = report["exit_code_lanes"]
        assert lanes["campaign"]["exit_code"] == 0
        assert lanes["campaign"]["soft_contract_warning"] is True
        assert lanes["job_exit_code"] == 0
        assert lanes["post_campaign_stage"]["exit_code"] == 0
        assert lanes["post_campaign_stage"]["status"] == "completed"

    def test_failed_campaign_stays_nonzero_through_finalizer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hard campaign failure must not be laundered to success by the finalizer."""
        from scripts.tools import slurm_job_finalize

        launcher_exit, campaign_root = self._run_launcher(
            tmp_path, monkeypatch, soft_warning=False, benchmark_success=False
        )
        assert launcher_exit != 0

        envelope = campaign_root / "reports" / "post_campaign_stage_status.json"
        assert envelope.is_file()

        finalize_output = tmp_path / "finalization.json"
        finalizer_exit = slurm_job_finalize.main(
            [
                "--repo-root",
                str(tmp_path),
                "--issue",
                "5244",
                "--job-id",
                "13274",
                "--job-state",
                "COMPLETED",
                "--expected-artifact",
                str(campaign_root / "reports" / "campaign_summary.json"),
                "--post-campaign-stage-status",
                str(envelope),
                "--output",
                str(finalize_output),
            ]
        )

        assert finalizer_exit == 1
        report = json.loads(finalize_output.read_text(encoding="utf-8"))
        assert report["classification"] == "failed"
        assert report["exit_code_lanes"]["campaign"]["exit_code"] == launcher_exit
        assert report["exit_code_lanes"]["job_exit_code"] == launcher_exit
