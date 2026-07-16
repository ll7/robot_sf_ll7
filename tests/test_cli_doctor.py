"""Tests for the top-level ``robot-sf doctor`` readiness command."""

from __future__ import annotations

from types import SimpleNamespace
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


def _write_quickstart_workspace(workspace: Path, relative_paths: tuple[str, ...]) -> None:
    """Create a minimal manifest-backed workspace for doctor tests.

    Writes ``examples/examples_manifest.yaml`` with a ``quickstart`` category and
    one example script per provided relative path (under ``examples/``), so the
    doctor's manifest-driven quickstart check has realistic inputs.
    """
    examples_root = workspace / "examples"
    examples_root.mkdir(parents=True)
    manifest_lines = [
        "version: 0.1.0",
        "categories:",
        "  - slug: quickstart",
        "    title: Quickstart",
        "    description: Test quickstarts",
        "    order: 1",
        "examples: []" if not relative_paths else "examples:",
    ]
    for index, relative_path in enumerate(relative_paths):
        manifest_lines.extend(
            [
                f"  - path: {relative_path}",
                f'    name: "Test Quickstart {index}"',
                "    summary: Test quickstart",
                "    category_slug: quickstart",
            ]
        )
        example_path = examples_root / relative_path
        example_path.parent.mkdir(parents=True, exist_ok=True)
        example_path.write_text("print('ok')\n", encoding="utf-8")
    (examples_root / "examples_manifest.yaml").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )


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
            "--skip-quickstart-smoke",
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
            "--skip-quickstart-smoke",
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
            "--skip-quickstart-smoke",
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
    _write_quickstart_workspace(workspace, ("quickstart/custom.py",))
    monkeypatch.setattr(doctor, "DEFAULT_WORKSPACE_ROOT", workspace)

    rc = main(
        [
            "doctor",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--skip-env-smoke",
            "--skip-quickstart-smoke",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "[WARN] model_artifacts" in out
    assert "Restore model artifacts" in out


def test_quickstart_smoke_uses_manifest_and_reports_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest-declared quickstarts must fail closed when execution fails."""
    workspace = tmp_path / "workspace"
    _write_quickstart_workspace(workspace, ("quickstart/custom.py",))

    def _fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(returncode=1, stdout="", stderr="smoke failed")

    monkeypatch.setattr(doctor.subprocess, "run", _fake_run)

    check = doctor._check_quickstart(workspace, tmp_path / "artifacts", run_smoke=True)

    assert check.status == "failed"
    assert check.details["present"] == ["examples/quickstart/custom.py"]
    assert check.details["smoke"] == "failed"
    assert check.details["failures"][0]["path"] == "examples/quickstart/custom.py"


def test_quickstart_check_fails_without_manifest_entries(tmp_path: Path) -> None:
    """An empty quickstart category must not be reported as runnable."""
    workspace = tmp_path / "workspace"
    _write_quickstart_workspace(workspace, ())

    check = doctor._check_quickstart(workspace, tmp_path / "artifacts", run_smoke=False)

    assert check.status == "failed"
    assert check.details["hint"] == "Add at least one manifest-declared quickstart example."


def test_optional_import_check_renders_remedy_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional import must carry an actionable install hint."""
    real_find_spec = doctor.importlib_util.find_spec

    def _fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "numpy":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(doctor.importlib_util, "find_spec", _fake_find_spec)

    check = doctor._check_optional_import("numpy")

    assert check.status == "missing_optional"
    assert check.details["available"] is False
    assert "hint" in check.details
    assert "uv sync --all-extras" in check.details["hint"]


def test_optional_binary_check_renders_remedy_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional binary must carry an actionable install hint."""
    real_which = doctor.shutil.which

    def _fake_which(name: str) -> str | None:
        if name == "ffmpeg":
            return None
        return real_which(name)

    monkeypatch.setattr(doctor.shutil, "which", _fake_which)

    check = doctor._check_binary("ffmpeg", required=False)

    assert check.status == "missing_optional"
    assert check.details["path"] is None
    assert "hint" in check.details
    assert "Install ffmpeg" in check.details["hint"]
