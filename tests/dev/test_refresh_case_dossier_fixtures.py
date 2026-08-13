"""Tests for the non-mutating Chapter 7 fixture refresh check."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/dev/refresh_case_dossier_fixtures.py"


@pytest.fixture(scope="module")
def refresh_module() -> ModuleType:
    """Load the script under test without requiring scripts/dev to be a package."""
    spec = importlib.util.spec_from_file_location("refresh_case_dossier_fixtures", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_mode_accepts_committed_fixtures(
    refresh_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The clean committed packages match a fresh canonical regeneration."""
    assert refresh_module.main(["--check"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["mode"] == "check"
    assert {folder for folder, _ in report["fixtures"]} == {
        "matched_seed118",
        "doorway_seeds113_114",
    }


def test_check_mode_reports_drift_and_source_digest(
    refresh_module: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed committed JSON file fails with hashes and current source provenance."""
    copied_root = tmp_path / "case_dossier_v1"
    shutil.copytree(refresh_module.FIXTURE_ROOT, copied_root)
    portfolio = copied_root / "matched_seed118" / "portfolio.json"
    portfolio.write_bytes(portfolio.read_bytes() + b"\n")
    monkeypatch.setattr(refresh_module, "FIXTURE_ROOT", copied_root)

    assert refresh_module.main(["--check"]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "drift"
    assert report["first_difference"]["path"] == "matched_seed118/portfolio.json"
    assert report["first_difference"]["status"] == "content_mismatch"
    assert report["generated_source_digests"]["matched_seed118"]["trace_package_sha256"]


def test_check_mode_does_not_modify_committed_fixtures(
    refresh_module: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The check path leaves every tracked JSON fixture byte-identical."""
    before = {
        path.relative_to(refresh_module.FIXTURE_ROOT).as_posix(): path.read_bytes()
        for path in refresh_module.FIXTURE_ROOT.rglob("*.json")
    }

    assert refresh_module.main(["--check"]) == 0
    capsys.readouterr()

    after = {
        path.relative_to(refresh_module.FIXTURE_ROOT).as_posix(): path.read_bytes()
        for path in refresh_module.FIXTURE_ROOT.rglob("*.json")
    }
    assert after == before
