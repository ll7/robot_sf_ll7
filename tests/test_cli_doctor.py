"""Tests for the top-level ``robot-sf doctor`` readiness command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from pathlib import Path

from robot_sf.benchmark import doctor
from robot_sf.cli import main


def _check_by_name(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return doctor checks keyed by name."""
    checks = payload["checks"]
    assert isinstance(checks, list)
    return {str(check["name"]): check for check in checks}


def test_robot_sf_doctor_happy_path_friendly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``robot-sf doctor`` should print a categorized pass/warn/fail report."""
    rc = main(
        [
            "doctor",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--skip-env-smoke",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("Robot SF environment check")
    # Required checks must be represented in the human report.
    assert "[PASS]" in out or "[WARN]" in out or "[FAIL]" in out
    # New capability: uv bootstrap + quickstart + model + optional extras.
    assert "uv_bootstrap" in out
    assert "quickstart" in out
    assert "model_artifacts" in out
    assert "optional_extras" in out


def test_robot_sf_doctor_json_format_roundtrip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``robot-sf doctor --format json`` should emit the JSON report."""
    import json

    rc = main(
        [
            "doctor",
            "--format",
            "json",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--skip-env-smoke",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    checks = _check_by_name(payload)
    assert payload["status"] in {"ok", "warning", "failed"}
    assert checks["uv_bootstrap"]["status"] == "ok"
    assert checks["optional_extras"]["status"] in {"ok", "missing_optional"}


def test_robot_sf_doctor_shows_remedy_for_missing_uv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing required binary must render a remedy hint in the friendly report."""
    real_which = doctor.shutil.which

    def _fake_which(name: str) -> str | None:
        if name == "uv":
            return None
        return real_which(name)

    monkeypatch.setattr(doctor.shutil, "which", _fake_which)

    rc = main(
        [
            "doctor",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--skip-env-smoke",
        ]
    )

    assert rc == 1
    out = capsys.readouterr().out
    assert "[FAIL] uv_bootstrap" in out
    # Remedy must point beginners at a concrete uv install path.
    assert "curl -LsSf https://astral.sh/uv/install.sh" in out


def test_robot_sf_doctor_reports_missing_model_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing bundled model artifacts should warn and suggest a remedy."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Ensure quickstart files exist so only the model artifact check warns.
    for rel in doctor.QUICKSTART_EXAMPLES:
        (workspace / rel).parent.mkdir(parents=True, exist_ok=True)
        (workspace / rel).write_text("# example\n", encoding="utf-8")
    monkeypatch.setattr(doctor, "DEFAULT_WORKSPACE_ROOT", workspace)

    rc = main(
        [
            "doctor",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--skip-env-smoke",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "[WARN] model_artifacts" in out
    assert "Restore model artifacts" in out
