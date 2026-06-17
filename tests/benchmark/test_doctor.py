"""Tests for the robot_sf_bench doctor diagnostics command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark import doctor
from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path


def _check_by_name(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return doctor checks keyed by name."""
    checks = payload["checks"]
    assert isinstance(checks, list)
    return {str(check["name"]): check for check in checks}


def test_doctor_cli_emits_json_and_can_skip_env_smoke(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Doctor CLI should emit pasteable JSON and support skipping env smoke."""
    rc = cli_main(
        [
            "doctor",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--skip-env-smoke",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    checks = _check_by_name(payload)
    assert payload["schema"] == "robot_sf_bench.doctor.v1"
    assert checks["python"]["status"] == "ok"
    assert checks["artifact_root"]["status"] == "ok"
    assert checks["env_smoke"]["status"] == "skipped"


def test_doctor_collect_report_marks_missing_optional_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional imports should warn instead of failing the command."""
    real_find_spec = doctor.importlib.util.find_spec

    def _fake_find_spec(name: str):
        if name == "pygame":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(doctor.importlib.util, "find_spec", _fake_find_spec)

    payload = doctor.collect_doctor_report(
        artifact_root=tmp_path / "artifacts",
        run_env_smoke=False,
    )

    checks = _check_by_name(payload)
    assert payload["status"] == "warning"
    assert checks["import:pygame"]["status"] == "missing_optional"
    assert doctor.doctor_exit_code(payload) == 0


def test_doctor_collect_report_fails_on_missing_required_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing required binaries should produce a failing doctor exit code."""

    def _fake_which(name: str) -> str | None:
        if name == "git":
            return None
        return f"/usr/bin/{name}"

    monkeypatch.setattr(doctor.shutil, "which", _fake_which)

    payload = doctor.collect_doctor_report(
        artifact_root=tmp_path / "artifacts",
        run_env_smoke=False,
    )

    checks = _check_by_name(payload)
    assert payload["status"] == "failed"
    assert checks["binary:git"]["status"] == "failed"
    assert doctor.doctor_exit_code(payload) == 1
