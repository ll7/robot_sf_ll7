"""Tests for the compare_coverage.py CLI tool contract (issue #8038)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

CLI_PATH = Path(__file__).resolve().parents[2] / "scripts" / "coverage" / "compare_coverage.py"


@pytest.fixture
def sample_coverage_payload() -> dict[str, object]:
    """Sample valid coverage.json structure matching coverage.py output."""
    return {
        "meta": {
            "version": "7.6.0",
            "timestamp": "2026-08-30T22:00:00",
        },
        "totals": {
            "covered_lines": 80,
            "num_statements": 100,
            "percent_covered": 80.0,
            "missing_lines": 20,
            "excluded_lines": 0,
        },
        "files": {
            "robot_sf/sim/simulator.py": {
                "summary": {
                    "covered_lines": 80,
                    "num_statements": 100,
                    "percent_covered": 80.0,
                    "missing_lines": 20,
                    "excluded_lines": 0,
                },
                "executed_lines": list(range(1, 81)),
                "missing_lines": list(range(81, 101)),
                "excluded_lines": [],
            }
        },
    }


def test_cli_help_displays_canonical_default_path() -> None:
    """Argparse help text must reflect output/coverage/coverage.json as the default."""
    res = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "output/coverage/coverage.json" in res.stdout


def test_cli_uses_canonical_default_path_when_omitted(
    tmp_path: Path, sample_coverage_payload: dict[str, object]
) -> None:
    """Omitting --current reads output/coverage/coverage.json in the current working directory."""
    cov_dir = tmp_path / "output" / "coverage"
    cov_dir.mkdir(parents=True, exist_ok=True)
    cov_file = cov_dir / "coverage.json"
    cov_file.write_text(json.dumps(sample_coverage_payload), encoding="utf-8")

    res = subprocess.run(
        [sys.executable, str(CLI_PATH), "--absolute-only", "--minimum-total", "75.0"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "Error:" not in res.stderr


def test_cli_explicit_current_path_overrides_default(
    tmp_path: Path, sample_coverage_payload: dict[str, object]
) -> None:
    """Explicit --current flag overrides default and loads the targeted file."""
    custom_file = tmp_path / "custom_report.json"
    custom_file.write_text(json.dumps(sample_coverage_payload), encoding="utf-8")

    res = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--current",
            str(custom_file),
            "--absolute-only",
            "--minimum-total",
            "75.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0


def test_cli_missing_current_file_exits_code_1(tmp_path: Path) -> None:
    """Missing current coverage file exits with code 1 and prints an error to stderr."""
    missing = tmp_path / "non_existent.json"
    res = subprocess.run(
        [sys.executable, str(CLI_PATH), "--current", str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 1
    assert "Error:" in res.stderr or "Error:" in res.stdout


def test_cli_malformed_current_json_exits_code_1(tmp_path: Path) -> None:
    """Malformed current JSON file exits with code 1 and prints an error."""
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text("{invalid json", encoding="utf-8")
    res = subprocess.run(
        [sys.executable, str(CLI_PATH), "--current", str(corrupted)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 1


def test_cli_absolute_only_without_minimum_total_exits_code_2() -> None:
    """Using --absolute-only without --minimum-total triggers argparse error (code 2)."""
    res = subprocess.run(
        [sys.executable, str(CLI_PATH), "--absolute-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 2
    assert "--absolute-only requires --minimum-total" in res.stderr


def test_cli_invalid_percentage_exits_code_2() -> None:
    """Passing a percentage outside 0..100 triggers argparse validation error (code 2)."""
    res = subprocess.run(
        [sys.executable, str(CLI_PATH), "--minimum-total", "150"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 2
    assert "invalid percentage" in res.stderr


def test_cli_minimum_total_pass_and_fail(
    tmp_path: Path, sample_coverage_payload: dict[str, object]
) -> None:
    """Minimum-total check passes when above floor and fails (code 1) when below floor."""
    cov_file = tmp_path / "coverage.json"
    cov_file.write_text(json.dumps(sample_coverage_payload), encoding="utf-8")

    # Pass case: 80.0% >= 75.0%
    res_pass = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--current",
            str(cov_file),
            "--absolute-only",
            "--minimum-total",
            "75.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_pass.returncode == 0

    # Fail case: 80.0% < 85.0%
    res_fail = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--current",
            str(cov_file),
            "--absolute-only",
            "--minimum-total",
            "85.0",
            "--format",
            "terminal",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_fail.returncode == 1
    assert "below the required 85.00% absolute floor" in res_fail.stderr


def test_cli_json_format_output(tmp_path: Path, sample_coverage_payload: dict[str, object]) -> None:
    """JSON format output includes minimum_total_coverage and absolute_floor_passed fields."""
    cov_file = tmp_path / "coverage.json"
    cov_file.write_text(json.dumps(sample_coverage_payload), encoding="utf-8")

    res = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--current",
            str(cov_file),
            "--absolute-only",
            "--minimum-total",
            "75.0",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    parsed = json.loads(res.stdout)
    assert parsed["current_coverage"] == 80.0
    assert parsed["minimum_total_coverage"] == 75.0
    assert parsed["absolute_floor_passed"] is True
