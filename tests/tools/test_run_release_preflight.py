"""Tests for the release preflight CLI runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from robot_sf.benchmark.release_preflight import SCHEMA_VERSION
from scripts.tools import run_release_preflight

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(path: Path, payload: object) -> None:
    """Write a YAML fixture, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _checklist(path: str) -> dict[str, object]:
    """Build a minimal release-preflight checklist fixture."""
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": "synthetic_release",
        "items": [
            {
                "item_id": "reproduction_record",
                "criterion": "reproduction",
                "check": "artifact_present",
                "path": path,
            }
        ],
    }


def test_run_release_preflight_passes_and_writes_reports(tmp_path: Path) -> None:
    """A satisfied checklist exits 0 and writes JSON plus Markdown reports."""
    repo_root = tmp_path / "repo"
    checklist_path = tmp_path / "checklist.yaml"
    out_json = tmp_path / "reports" / "release_preflight_report.json"
    out_md = tmp_path / "reports" / "release_preflight_report.md"
    (repo_root / "docs/context/evidence/release_july_2026").mkdir(parents=True)
    (repo_root / "docs/context/evidence/release_july_2026/reproduction_record.md").write_text(
        "synthetic reproduction record\n",
        encoding="utf-8",
    )
    _write_yaml(
        checklist_path,
        _checklist("docs/context/evidence/release_july_2026/reproduction_record.md"),
    )

    exit_code = run_release_preflight.main(
        [
            "--checklist",
            str(checklist_path),
            "--repo-root",
            str(repo_root),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["summary"] == {"blocked": 0, "complete": 1, "total": 1}
    assert "Status: **passed**" in out_md.read_text(encoding="utf-8")


def test_run_release_preflight_blocked_exits_2_and_writes_reports(tmp_path: Path) -> None:
    """A blocked checklist exits 2 while preserving evaluator reports."""
    checklist_path = tmp_path / "checklist.yaml"
    out_json = tmp_path / "release_preflight_report.json"
    out_md = tmp_path / "release_preflight_report.md"
    _write_yaml(checklist_path, _checklist("docs/context/evidence/missing.md"))

    exit_code = run_release_preflight.main(
        [
            "--checklist",
            str(checklist_path),
            "--repo-root",
            str(tmp_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert exit_code == 2
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["items"][0]["status"] == "blocked"
    assert "missing" in payload["items"][0]["gaps"][0]
    assert "Status: **blocked**" in out_md.read_text(encoding="utf-8")


def test_run_release_preflight_malformed_checklist_exits_1(tmp_path: Path, capsys) -> None:
    """A malformed checklist exits 1 and cannot produce a false pass."""
    checklist_path = tmp_path / "bad.yaml"
    out_json = tmp_path / "release_preflight_report.json"
    out_md = tmp_path / "release_preflight_report.md"
    _write_yaml(
        checklist_path,
        {
            "schema_version": "wrong",
            "release_id": "synthetic_release",
            "items": [],
        },
    )

    exit_code = run_release_preflight.main(
        [
            "--checklist",
            str(checklist_path),
            "--repo-root",
            str(tmp_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert exit_code == 1
    assert not out_json.exists()
    assert not out_md.exists()
    assert "release preflight failed:" in capsys.readouterr().err
