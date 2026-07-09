"""Characterization wave 3 tests for scripts/tools/run_camera_ready_benchmark.py.

Focuses on argument/manifest handling at dry-run level (preflight mode).
NEW tests only, zero production changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools import run_camera_ready_benchmark

# ---------------------------------------------------------------------------
# Parser argument validation
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Tests for _build_parser argument validation."""

    def test_parser_requires_config(self) -> None:
        """--config is required."""
        parser = run_camera_ready_benchmark._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_defaults(self) -> None:
        """Parser defaults are correct."""
        parser = run_camera_ready_benchmark._build_parser()
        args = parser.parse_args(["--config", "test.yaml"])
        assert args.config == Path("test.yaml")
        assert args.output_root is None
        assert args.label is None
        assert args.campaign_id is None
        assert args.skip_publication_bundle is False
        assert args.mode == "run"
        assert args.checkpoint_preflight_mode == "metadata_only"
        assert args.checkpoint_cache_dir is None
        assert args.checkpoint_registry_path is None
        assert args.log_level == "INFO"
        assert args.arm_isolation == "in_process"

    def test_parser_mode_choices(self) -> None:
        """--mode accepts only 'run' and 'preflight'."""
        parser = run_camera_ready_benchmark._build_parser()
        args = parser.parse_args(["--config", "test.yaml", "--mode", "preflight"])
        assert args.mode == "preflight"

        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "test.yaml", "--mode", "invalid"])

    def test_parser_checkpoint_preflight_mode_choices(self) -> None:
        """--checkpoint-preflight-mode accepts only valid choices."""
        parser = run_camera_ready_benchmark._build_parser()
        args = parser.parse_args(
            ["--config", "test.yaml", "--checkpoint-preflight-mode", "enforced_staged"]
        )
        assert args.checkpoint_preflight_mode == "enforced_staged"

        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "test.yaml", "--checkpoint-preflight-mode", "invalid"])

    def test_parser_log_level_choices(self) -> None:
        """--log-level accepts only valid choices."""
        parser = run_camera_ready_benchmark._build_parser()
        args = parser.parse_args(["--config", "test.yaml", "--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "test.yaml", "--log-level", "INVALID"])

    def test_parser_arm_isolation_choices(self) -> None:
        """--arm-isolation accepts only valid choices."""
        parser = run_camera_ready_benchmark._build_parser()
        args = parser.parse_args(["--config", "test.yaml", "--arm-isolation", "subprocess"])
        assert args.arm_isolation == "subprocess"

        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "test.yaml", "--arm-isolation", "invalid"])

    def test_parser_skip_publication_bundle_flag(self) -> None:
        """--skip-publication-bundle flag is parsed correctly."""
        parser = run_camera_ready_benchmark._build_parser()
        args = parser.parse_args(["--config", "test.yaml", "--skip-publication-bundle"])
        assert args.skip_publication_bundle is True

    def test_parser_all_options(self) -> None:
        """All options can be set simultaneously."""
        parser = run_camera_ready_benchmark._build_parser()
        args = parser.parse_args(
            [
                "--config",
                "my_config.yaml",
                "--output-root",
                "/tmp/output",
                "--label",
                "test_label",
                "--campaign-id",
                "camp_123",
                "--skip-publication-bundle",
                "--mode",
                "preflight",
                "--checkpoint-preflight-mode",
                "enforced_staged",
                "--checkpoint-cache-dir",
                "/tmp/cache",
                "--checkpoint-registry-path",
                "/tmp/registry",
                "--log-level",
                "WARNING",
                "--arm-isolation",
                "subprocess",
            ]
        )
        assert args.config == Path("my_config.yaml")
        assert args.output_root == Path("/tmp/output")
        assert args.label == "test_label"
        assert args.campaign_id == "camp_123"
        assert args.skip_publication_bundle is True
        assert args.mode == "preflight"
        assert args.checkpoint_preflight_mode == "enforced_staged"
        assert args.checkpoint_cache_dir == Path("/tmp/cache")
        assert args.checkpoint_registry_path == Path("/tmp/registry")
        assert args.log_level == "WARNING"
        assert args.arm_isolation == "subprocess"


# ---------------------------------------------------------------------------
# Preflight manifest handling
# ---------------------------------------------------------------------------


class TestPreflightManifestHandling:
    """Tests for preflight manifest handling."""

    def test_preflight_manifest_keys(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """Preflight mode returns all required manifest keys."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        sentinel_cfg = object()

        def _fake_load(path: Path):
            return sentinel_cfg

        def _fake_preflight(cfg, **kwargs):
            return {
                "campaign_id": "cid",
                "campaign_root": tmp_path / "out",
                "validate_config_path": tmp_path / "out" / "validate.json",
                "preview_scenarios_path": tmp_path / "out" / "preview.json",
                "matrix_summary_json_path": tmp_path / "out" / "summary.json",
                "matrix_summary_csv_path": tmp_path / "out" / "summary.csv",
                "amv_coverage_json_path": tmp_path / "out" / "amv.json",
                "amv_coverage_md_path": tmp_path / "out" / "amv.md",
                "comparability_json_path": tmp_path / "out" / "comp.json",
                "comparability_md_path": tmp_path / "out" / "comp.md",
            }

        monkeypatch.setattr(run_camera_ready_benchmark, "load_campaign_config", _fake_load)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", _fake_preflight
        )
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "run_campaign",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        rc = run_camera_ready_benchmark.main(["--config", str(config_path), "--mode", "preflight"])
        assert rc == 0

        # Verify the emitted manifest payload actually carries every required key
        # (rc == 0 alone would pass even if main() dropped keys from the payload).
        payload = json.loads(capsys.readouterr().out)
        out = tmp_path / "out"
        assert payload == {
            "campaign_id": "cid",
            "campaign_root": str(out),
            "validate_config_path": str(out / "validate.json"),
            "preview_scenarios_path": str(out / "preview.json"),
            "matrix_summary_json": str(out / "summary.json"),
            "matrix_summary_csv": str(out / "summary.csv"),
            "amv_coverage_json": str(out / "amv.json"),
            "amv_coverage_md": str(out / "amv.md"),
            "comparability_json": str(out / "comp.json"),
            "comparability_md": str(out / "comp.md"),
        }

    def test_preflight_manifest_campaign_id_forwarded(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """Preflight forwards campaign_id to the preflight function."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        sentinel_cfg = object()
        received_kwargs: dict = {}

        def _fake_load(path: Path):
            return sentinel_cfg

        def _fake_preflight(cfg, **kwargs):
            received_kwargs.update(kwargs)
            return {
                "campaign_id": "cid",
                "campaign_root": tmp_path / "out",
                "validate_config_path": tmp_path / "out" / "v.json",
                "preview_scenarios_path": tmp_path / "out" / "p.json",
                "matrix_summary_json_path": tmp_path / "out" / "s.json",
                "matrix_summary_csv_path": tmp_path / "out" / "s.csv",
                "amv_coverage_json_path": tmp_path / "out" / "a.json",
                "amv_coverage_md_path": tmp_path / "out" / "a.md",
                "comparability_json_path": None,
                "comparability_md_path": None,
            }

        monkeypatch.setattr(run_camera_ready_benchmark, "load_campaign_config", _fake_load)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", _fake_preflight
        )
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "run_campaign",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        rc = run_camera_ready_benchmark.main(
            [
                "--config",
                str(config_path),
                "--mode",
                "preflight",
                "--campaign-id",
                "my_campaign",
            ]
        )
        assert rc == 0
        assert received_kwargs.get("campaign_id") == "my_campaign"

    def test_preflight_manifest_output_root_forwarded(self, tmp_path: Path, monkeypatch) -> None:
        """Preflight forwards output_root to the preflight function."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        sentinel_cfg = object()
        received_kwargs: dict = {}

        def _fake_load(path: Path):
            return sentinel_cfg

        def _fake_preflight(cfg, **kwargs):
            received_kwargs.update(kwargs)
            return {
                "campaign_id": "cid",
                "campaign_root": tmp_path / "out",
                "validate_config_path": tmp_path / "out" / "v.json",
                "preview_scenarios_path": tmp_path / "out" / "p.json",
                "matrix_summary_json_path": tmp_path / "out" / "s.json",
                "matrix_summary_csv_path": tmp_path / "out" / "s.csv",
                "amv_coverage_json_path": tmp_path / "out" / "a.json",
                "amv_coverage_md_path": tmp_path / "out" / "a.md",
                "comparability_json_path": None,
                "comparability_md_path": None,
            }

        monkeypatch.setattr(run_camera_ready_benchmark, "load_campaign_config", _fake_load)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", _fake_preflight
        )
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "run_campaign",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        custom_root = tmp_path / "custom_output"
        rc = run_camera_ready_benchmark.main(
            [
                "--config",
                str(config_path),
                "--mode",
                "preflight",
                "--output-root",
                str(custom_root),
            ]
        )
        assert rc == 0
        assert received_kwargs.get("output_root") == custom_root

    def test_preflight_manifest_label_forwarded(self, tmp_path: Path, monkeypatch) -> None:
        """Preflight forwards label to the preflight function."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        sentinel_cfg = object()
        received_kwargs: dict = {}

        def _fake_load(path: Path):
            return sentinel_cfg

        def _fake_preflight(cfg, **kwargs):
            received_kwargs.update(kwargs)
            return {
                "campaign_id": "cid",
                "campaign_root": tmp_path / "out",
                "validate_config_path": tmp_path / "out" / "v.json",
                "preview_scenarios_path": tmp_path / "out" / "p.json",
                "matrix_summary_json_path": tmp_path / "out" / "s.json",
                "matrix_summary_csv_path": tmp_path / "out" / "s.csv",
                "amv_coverage_json_path": tmp_path / "out" / "a.json",
                "amv_coverage_md_path": tmp_path / "out" / "a.md",
                "comparability_json_path": None,
                "comparability_md_path": None,
            }

        monkeypatch.setattr(run_camera_ready_benchmark, "load_campaign_config", _fake_load)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", _fake_preflight
        )
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "run_campaign",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        rc = run_camera_ready_benchmark.main(
            [
                "--config",
                str(config_path),
                "--mode",
                "preflight",
                "--label",
                "my_label",
            ]
        )
        assert rc == 0
        assert received_kwargs.get("label") == "my_label"

    def test_preflight_manifest_checkpoint_options_forwarded(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Preflight forwards checkpoint options to the preflight function."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        sentinel_cfg = object()
        received_kwargs: dict = {}

        def _fake_load(path: Path):
            return sentinel_cfg

        def _fake_preflight(cfg, **kwargs):
            received_kwargs.update(kwargs)
            return {
                "campaign_id": "cid",
                "campaign_root": tmp_path / "out",
                "validate_config_path": tmp_path / "out" / "v.json",
                "preview_scenarios_path": tmp_path / "out" / "p.json",
                "matrix_summary_json_path": tmp_path / "out" / "s.json",
                "matrix_summary_csv_path": tmp_path / "out" / "s.csv",
                "amv_coverage_json_path": tmp_path / "out" / "a.json",
                "amv_coverage_md_path": tmp_path / "out" / "a.md",
                "comparability_json_path": None,
                "comparability_md_path": None,
            }

        monkeypatch.setattr(run_camera_ready_benchmark, "load_campaign_config", _fake_load)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", _fake_preflight
        )
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "run_campaign",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        rc = run_camera_ready_benchmark.main(
            [
                "--config",
                str(config_path),
                "--mode",
                "preflight",
                "--checkpoint-preflight-mode",
                "enforced_staged",
                "--checkpoint-cache-dir",
                "/tmp/cache",
                "--checkpoint-registry-path",
                "/tmp/registry",
            ]
        )
        assert rc == 0
        assert received_kwargs.get("checkpoint_preflight_mode") == "enforced_staged"
        assert received_kwargs.get("checkpoint_cache_dir") == Path("/tmp/cache")
        assert received_kwargs.get("checkpoint_registry_path") == Path("/tmp/registry")

    def test_preflight_manifest_comparability_fields_optional(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """Preflight handles None comparability fields gracefully."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        sentinel_cfg = object()

        def _fake_load(path: Path):
            return sentinel_cfg

        def _fake_preflight(cfg, **kwargs):
            return {
                "campaign_id": "cid",
                "campaign_root": tmp_path / "out",
                "validate_config_path": tmp_path / "out" / "v.json",
                "preview_scenarios_path": tmp_path / "out" / "p.json",
                "matrix_summary_json_path": tmp_path / "out" / "s.json",
                "matrix_summary_csv_path": tmp_path / "out" / "s.csv",
                "amv_coverage_json_path": tmp_path / "out" / "a.json",
                "amv_coverage_md_path": tmp_path / "out" / "a.md",
                "comparability_json_path": None,
                "comparability_md_path": None,
            }

        monkeypatch.setattr(run_camera_ready_benchmark, "load_campaign_config", _fake_load)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", _fake_preflight
        )
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "run_campaign",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        rc = run_camera_ready_benchmark.main(["--config", str(config_path), "--mode", "preflight"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["comparability_json"] is None
        assert payload["comparability_md"] is None


# ---------------------------------------------------------------------------
# Run mode manifest handling
# ---------------------------------------------------------------------------


class TestRunModeManifestHandling:
    """Tests for run mode manifest handling."""

    def test_run_mode_forwards_all_options(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """Run mode forwards all CLI options to run_campaign."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        sentinel_cfg = object()
        received_kwargs: dict = {}

        def _fake_load(path: Path):
            return sentinel_cfg

        def _fake_run(cfg, **kwargs):
            received_kwargs.update(kwargs)
            return {
                "campaign_id": "cid",
                "campaign_root": str(tmp_path / "out"),
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

        monkeypatch.setattr(run_camera_ready_benchmark, "load_campaign_config", _fake_load)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", lambda *a, **kw: {}
        )
        monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run)

        rc = run_camera_ready_benchmark.main(
            [
                "--config",
                str(config_path),
                "--campaign-id",
                "my_camp",
                "--skip-publication-bundle",
                "--arm-isolation",
                "subprocess",
            ]
        )
        assert rc == 0
        assert received_kwargs.get("campaign_id") == "my_camp"
        assert received_kwargs.get("skip_publication_bundle") is True
        assert received_kwargs.get("arm_isolation") == "subprocess"

    def test_run_mode_invoked_command_contains_args(self, tmp_path: Path, monkeypatch) -> None:
        """Run mode constructs invoked_command from CLI arguments."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        sentinel_cfg = object()
        received_kwargs: dict = {}

        def _fake_load(path: Path):
            return sentinel_cfg

        def _fake_run(cfg, **kwargs):
            received_kwargs.update(kwargs)
            return {
                "campaign_id": "cid",
                "campaign_root": str(tmp_path / "out"),
                "benchmark_success": True,
                "successful_runs": 1,
                "non_success_runs": 0,
                "total_runs": 1,
            }

        monkeypatch.setattr(run_camera_ready_benchmark, "load_campaign_config", _fake_load)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", lambda *a, **kw: {}
        )
        monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run)

        rc = run_camera_ready_benchmark.main(
            [
                "--config",
                str(config_path),
                "--campaign-id",
                "test_camp",
            ]
        )
        assert rc == 0
        invoked = received_kwargs.get("invoked_command", "")
        assert "test_camp" in invoked


# ---------------------------------------------------------------------------
# OrcaRvo2PreflightError handling
# ---------------------------------------------------------------------------


class TestOrcaPreflightErrorHandling:
    """Tests for OrcaRvo2PreflightError handling in main."""

    def test_orca_error_produces_structured_payload(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """OrcaRvo2PreflightError is caught and produces structured JSON."""
        from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError

        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        sentinel_cfg = object()

        def _fake_load(path: Path):
            return sentinel_cfg

        def _fake_run(cfg, **kwargs):
            raise OrcaRvo2PreflightError("rvo2 not installed")

        monkeypatch.setattr(run_camera_ready_benchmark, "load_campaign_config", _fake_load)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", lambda *a, **kw: {}
        )
        monkeypatch.setattr(run_camera_ready_benchmark, "run_campaign", _fake_run)

        rc = run_camera_ready_benchmark.main(["--config", str(config_path)])
        assert rc == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "orca_preflight_failed"
        assert payload["benchmark_success"] is False
        assert payload["exit_code"] == 2
        assert "rvo2 not installed" in payload["status_reason"]

    def test_orca_error_in_preflight_mode(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """OrcaRvo2PreflightError in preflight mode produces structured JSON."""
        from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError

        config_path = tmp_path / "config.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        sentinel_cfg = object()

        def _fake_load(path: Path):
            return sentinel_cfg

        def _fake_preflight(cfg, **kwargs):
            raise OrcaRvo2PreflightError("rvo2 not available")

        monkeypatch.setattr(run_camera_ready_benchmark, "load_campaign_config", _fake_load)
        monkeypatch.setattr(
            run_camera_ready_benchmark, "prepare_campaign_preflight", _fake_preflight
        )
        monkeypatch.setattr(
            run_camera_ready_benchmark,
            "run_campaign",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable")),
        )

        rc = run_camera_ready_benchmark.main(["--config", str(config_path), "--mode", "preflight"])
        assert rc == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["mode"] == "preflight"
        assert payload["status"] == "orca_preflight_failed"
