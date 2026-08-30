"""CLI tests for scripts/coverage/compare_coverage.py."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from scripts.coverage.compare_coverage import main

if TYPE_CHECKING:
    from pathlib import Path


def _make_cov_json(percent: float) -> dict:
    return {
        "meta": {"version": "7.0.0"},
        "totals": {
            "covered_lines": int(percent * 10),
            "num_statements": 1000,
            "percent_covered": percent,
            "percent_covered_display": f"{percent:.2f}",
            "missing_lines": 1000 - int(percent * 10),
            "excluded_lines": 0,
        },
        "files": {},
    }


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI help displays default path output/coverage/coverage.json."""
    with (
        patch.object(sys, "argv", ["compare_coverage.py", "--help"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "output/coverage/coverage.json" in out


def test_cli_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI reads output/coverage/coverage.json by default."""
    cov_dir = tmp_path / "output" / "coverage"
    cov_dir.mkdir(parents=True, exist_ok=True)
    cov_file = cov_dir / "coverage.json"
    cov_file.write_text(json.dumps(_make_cov_json(90.0)), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    with patch.object(
        sys, "argv", ["compare_coverage.py", "--absolute-only", "--minimum-total", "85.0"]
    ):
        exit_code = main()

    assert exit_code == 0


def test_cli_explicit_current_path(tmp_path: Path) -> None:
    """CLI accepts explicit --current path."""
    custom = tmp_path / "custom_cov.json"
    custom.write_text(json.dumps(_make_cov_json(88.0)), encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "compare_coverage.py",
            "--current",
            str(custom),
            "--absolute-only",
            "--minimum-total",
            "80.0",
        ],
    ):
        exit_code = main()

    assert exit_code == 0


def test_cli_missing_current_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing current coverage file exits 1 with an actionable error."""
    missing = tmp_path / "nonexistent.json"
    with patch.object(
        sys,
        "argv",
        [
            "compare_coverage.py",
            "--current",
            str(missing),
            "--absolute-only",
            "--minimum-total",
            "80.0",
        ],
    ):
        exit_code = main()

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Error:" in err


def test_cli_malformed_current_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Malformed current JSON exits 1 with an actionable error."""
    malformed = tmp_path / "bad.json"
    malformed.write_text("not json", encoding="utf-8")
    with patch.object(
        sys,
        "argv",
        [
            "compare_coverage.py",
            "--current",
            str(malformed),
            "--absolute-only",
            "--minimum-total",
            "80.0",
        ],
    ):
        exit_code = main()

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Error:" in err


def test_cli_absolute_only_requires_minimum_total(capsys: pytest.CaptureFixture[str]) -> None:
    """--absolute-only without --minimum-total exits 2 with argparse error."""
    with (
        patch.object(sys, "argv", ["compare_coverage.py", "--absolute-only"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--absolute-only requires --minimum-total" in err


def test_cli_invalid_percentage(capsys: pytest.CaptureFixture[str]) -> None:
    """--minimum-total with invalid percentage exits 2."""
    with (
        patch.object(sys, "argv", ["compare_coverage.py", "--minimum-total", "150.0"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "invalid percentage value: '150.0'" in err


def test_cli_minimum_total_floor_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When coverage is below minimum total, CLI reports failure and exits 1."""
    cov_file = tmp_path / "coverage.json"
    cov_file.write_text(json.dumps(_make_cov_json(75.0)), encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "compare_coverage.py",
            "--current",
            str(cov_file),
            "--absolute-only",
            "--minimum-total",
            "80.0",
            "--format",
            "terminal",
        ],
    ):
        exit_code = main()

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Total coverage 75.00% is below the required 80.00% absolute floor." in err


def test_cli_json_format_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """JSON output format contains minimum_total_coverage and absolute_floor_passed."""
    cov_file = tmp_path / "coverage.json"
    cov_file.write_text(json.dumps(_make_cov_json(85.5)), encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "compare_coverage.py",
            "--current",
            str(cov_file),
            "--absolute-only",
            "--minimum-total",
            "80.0",
            "--format",
            "json",
        ],
    ):
        exit_code = main()

    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["current_coverage"] == 85.5
    assert data["minimum_total_coverage"] == 80.0
    assert data["absolute_floor_passed"] is True
