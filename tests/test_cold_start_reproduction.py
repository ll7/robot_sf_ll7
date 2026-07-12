#!/usr/bin/env python3
"""Tests for cold-start reproduction script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add the scripts/repro directory to path so we can import the module
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "repro"))

from cold_start_reproduction import (
    MINIMAL_SUBSET,
    collect_environment_info,
    generate_reproduction_report,
    verify_artifact_checksums,
)


class TestCollectEnvironmentInfo:
    """Tests for environment info collection."""

    def test_returns_required_fields(self):
        """Test that all required fields are present."""
        info = collect_environment_info()

        assert "platform" in info
        assert "python_version" in info
        assert "architecture" in info
        assert "timestamp_utc" in info

    def test_platform_is_string(self):
        """Test that platform is a non-empty string."""
        info = collect_environment_info()
        assert isinstance(info["platform"], str)
        assert len(info["platform"]) > 0

    def test_python_version_format(self):
        """Test that python_version has expected format."""
        info = collect_environment_info()
        # Should be like "3.12.0"
        parts = info["python_version"].split(".")
        assert len(parts) >= 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()


class TestVerifyArtifactChecksums:
    """Tests for artifact checksum verification."""

    def test_returns_dict_for_existing_files(self, tmp_path: Path):
        """Test that checksums are returned for existing files."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        # Create the expected directory structure
        (tmp_path / "configs" / "scenarios").mkdir(parents=True)
        (tmp_path / "maps" / "svg_maps").mkdir(parents=True)

        results = verify_artifact_checksums(tmp_path)

        # Should return a dict
        assert isinstance(results, dict)

    def test_missing_files_marked_correctly(self, tmp_path: Path):
        """Test that missing files are marked as missing."""
        results = verify_artifact_checksums(tmp_path)

        # All files should be missing in empty directory
        for info in results.values():
            assert info["status"] == "missing"
            assert info["sha256"] == ""
            assert info["bytes"] == "0"

    def test_existing_file_has_hash(self, tmp_path: Path):
        """Test that existing files have a valid hash."""
        # Create pyproject.toml
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'\n")

        results = verify_artifact_checksums(tmp_path)

        # pyproject.toml should be present
        assert results["pyproject.toml"]["status"] == "present"
        assert len(results["pyproject.toml"]["sha256"]) == 64  # SHA-256 hex length


class TestMinimalSubset:
    """Tests for minimal subset configuration."""

    def test_has_required_keys(self):
        """Test that minimal subset has all required keys."""
        required_keys = ["matrix", "algo", "horizon", "repeats", "workers", "benchmark_profile"]
        for key in required_keys:
            assert key in MINIMAL_SUBSET, f"Missing key: {key}"

    def test_matrix_path_format(self):
        """Test that matrix path is valid."""
        assert MINIMAL_SUBSET["matrix"].endswith(".yaml")
        assert "/" in MINIMAL_SUBSET["matrix"]  # Should be a path

    def test_numeric_values_are_positive(self):
        """Test that numeric values are positive."""
        assert MINIMAL_SUBSET["horizon"] > 0
        assert MINIMAL_SUBSET["repeats"] > 0
        assert MINIMAL_SUBSET["workers"] > 0


class TestGenerateReproductionReport:
    """Tests for report generation."""

    def test_report_has_required_sections(self, tmp_path: Path):
        """Test that report has all required sections."""
        env_info = {"platform": "test", "python_version": "3.12.0"}
        checksums = {"test.txt": {"status": "present", "sha256": "abc", "bytes": "10"}}
        build_result = {"status": "success", "wall_time_s": 10.0}
        benchmark_result = {"status": "success", "manifest": {"episode_count": 2}}
        output_path = tmp_path / "report.json"

        report = generate_reproduction_report(
            env_info, checksums, build_result, benchmark_result, output_path
        )

        assert "schema" in report
        assert "environment" in report
        assert "checksum_verification" in report
        assert "build_result" in report
        assert "benchmark_result" in report
        assert "limitations" in report
        assert "instruction_gaps_found" in report

    def test_report_written_to_file(self, tmp_path: Path):
        """Test that report is written to the specified file."""
        env_info = {"platform": "test"}
        checksums: dict[str, dict[str, str]] = {}
        build_result = {"status": "success"}
        benchmark_result = {"status": "success"}
        output_path = tmp_path / "report.json"

        generate_reproduction_report(
            env_info, checksums, build_result, benchmark_result, output_path
        )

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["schema"] == "robot-sf-cold-start-reproduction-report.v1"

    def test_identifies_build_failure_gap(self, tmp_path: Path):
        """Test that build failures are identified as instruction gaps."""
        env_info = {"platform": "test"}
        checksums: dict[str, dict[str, str]] = {}
        build_result = {"status": "failed", "step": "venv_creation"}
        benchmark_result = {"status": "skipped"}
        output_path = tmp_path / "report.json"

        report = generate_reproduction_report(
            env_info, checksums, build_result, benchmark_result, output_path
        )

        assert len(report["instruction_gaps_found"]) > 0
        assert any("Build failed" in gap for gap in report["instruction_gaps_found"])

    def test_identifies_benchmark_failure_gap(self, tmp_path: Path):
        """Test that benchmark failures are identified as instruction gaps."""
        env_info = {"platform": "test"}
        checksums: dict[str, dict[str, str]] = {}
        build_result = {"status": "success"}
        benchmark_result = {"status": "failed", "manifest": {"benchmark_exit_code": 1}}
        output_path = tmp_path / "report.json"

        report = generate_reproduction_report(
            env_info, checksums, build_result, benchmark_result, output_path
        )

        assert len(report["instruction_gaps_found"]) > 0
        assert any("Benchmark failed" in gap for gap in report["instruction_gaps_found"])

    def test_identifies_missing_artifacts_gap(self, tmp_path: Path):
        """Test that missing artifacts are identified as instruction gaps."""
        env_info = {"platform": "test"}
        checksums = {"missing.txt": {"status": "missing", "sha256": "", "bytes": "0"}}
        build_result = {"status": "success"}
        benchmark_result = {"status": "success"}
        output_path = tmp_path / "report.json"

        report = generate_reproduction_report(
            env_info, checksums, build_result, benchmark_result, output_path
        )

        assert len(report["instruction_gaps_found"]) > 0
        assert any("Missing" in gap for gap in report["instruction_gaps_found"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
