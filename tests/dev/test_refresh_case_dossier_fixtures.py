"""Tests for deterministic case-dossier fixture refresh and drift checking."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "dev"
sys.path.insert(0, str(SCRIPT_DIR))
import refresh_case_dossier_fixtures as refresh  # noqa: E402


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    """Capture all regular files below a fixture package root."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_check_mode_reports_clean_fixture_packages(capsys) -> None:
    """The committed production-shaped packages match fresh builder output."""
    assert refresh.main(["--check"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "check"
    assert report["status"] == "ok"
    assert [fixture["fixture"] for fixture in report["fixtures"]] == [
        "matched_seed118",
        "doorway_seeds113_114",
    ]
    assert all(len(fixture["source_digest"]) == 64 for fixture in report["fixtures"])
    assert all(fixture["status"] == "ok" for fixture in report["fixtures"])


def test_check_mode_reports_first_difference_and_source_digest(
    tmp_path, monkeypatch, capsys
) -> None:
    """Drift identifies the first changed file and leaves the copied tree untouched."""
    fixture_root = tmp_path / "case_dossier_v1"
    shutil.copytree(refresh.FIXTURE_ROOT, fixture_root)
    input_path = fixture_root / "matched_seed118" / "input.json"
    input_path.write_bytes(input_path.read_bytes() + b" ")
    before = _tree_snapshot(fixture_root)
    monkeypatch.setattr(refresh, "FIXTURE_ROOT", fixture_root)

    assert refresh.main(["--check"]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "drift"
    matched = report["fixtures"][0]
    assert matched["status"] == "drift"
    assert matched["first_differing_path"].endswith("matched_seed118/input.json")
    assert len(matched["source_digest"]) == 64
    assert len(matched["tracked_digest"]) == 64
    assert matched["source_digest"] != matched["tracked_digest"]
    assert _tree_snapshot(fixture_root) == before


def test_check_mode_is_non_mutating_for_clean_tree(tmp_path, monkeypatch, capsys) -> None:
    """A clean check does not rewrite or add files in the checked tree."""
    fixture_root = tmp_path / "case_dossier_v1"
    shutil.copytree(refresh.FIXTURE_ROOT, fixture_root)
    before = _tree_snapshot(fixture_root)
    monkeypatch.setattr(refresh, "FIXTURE_ROOT", fixture_root)

    assert refresh.main(["--check"]) == 0
    capsys.readouterr()

    assert _tree_snapshot(fixture_root) == before
