"""Tests for the v0.1 benchmark publication prerequisites preflight (epic #2910)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.validation import check_benchmark_v0_1_publication_prerequisites as preflight

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = preflight.SCHEMA_VERSION


def _write_checklist(path: Path, prerequisites: list[dict]) -> Path:
    """Write a synthetic checklist YAML and return its path."""
    payload = {
        "schema_version": SCHEMA,
        "release_target": "paper-benchmark-v0.1.0",
        "epic": 2910,
        "prerequisites": prerequisites,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _touch(repo_root: Path, rel: str) -> None:
    """Create an empty file (and parents) under the synthetic repo root."""
    target = repo_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")


def test_all_present_passes(tmp_path: Path) -> None:
    """When every required path exists the report is satisfied and main returns 0."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _touch(repo, "a/present.py")
    checklist = _write_checklist(
        tmp_path / "checklist.yaml",
        [{"id": "a", "category": "contract", "required": True, "paths": ["a/present.py"]}],
    )
    report = preflight.build_report(checklist, repo)

    assert report["prerequisites_satisfied"] is True
    assert report["required_missing"] == []
    assert report["present"] == report["total"] == 1
    assert preflight.main(["--checklist", str(checklist), "--repo-root", str(repo)]) == 0


def test_missing_required_fails_closed(tmp_path: Path) -> None:
    """A missing required path reports the id, is unsatisfied, and main returns 1."""
    repo = tmp_path / "repo"
    repo.mkdir()
    checklist = _write_checklist(
        tmp_path / "checklist.yaml",
        [{"id": "gone", "category": "release", "required": True, "paths": ["nope.yaml"]}],
    )
    report = preflight.build_report(checklist, repo)

    assert report["prerequisites_satisfied"] is False
    assert report["required_missing"] == ["gone"]
    assert report["prerequisites"][0]["missing_paths"] == ["nope.yaml"]
    assert preflight.main(["--checklist", str(checklist), "--repo-root", str(repo)]) == 1


def test_missing_optional_does_not_fail(tmp_path: Path) -> None:
    """A missing optional prerequisite is reported but does not fail the preflight."""
    repo = tmp_path / "repo"
    repo.mkdir()
    checklist = _write_checklist(
        tmp_path / "checklist.yaml",
        [{"id": "opt", "category": "seed", "required": False, "paths": ["maybe.yaml"]}],
    )
    report = preflight.build_report(checklist, repo)

    assert report["prerequisites_satisfied"] is True
    assert report["optional_missing"] == ["opt"]
    assert preflight.main(["--checklist", str(checklist), "--repo-root", str(repo)]) == 0


def test_multi_path_requires_all(tmp_path: Path) -> None:
    """An item with multiple paths is present only when every path exists."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _touch(repo, "one.py")
    checklist = _write_checklist(
        tmp_path / "checklist.yaml",
        [
            {
                "id": "pair",
                "category": "claim_matrix",
                "required": True,
                "paths": ["one.py", "two.py"],
            }
        ],
    )
    report = preflight.build_report(checklist, repo)

    assert report["prerequisites_satisfied"] is False
    assert report["prerequisites"][0]["missing_paths"] == ["two.py"]


def test_report_always_carries_claim_boundary(tmp_path: Path) -> None:
    """The report surfaces an explicit presence-only boundary string."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _touch(repo, "x.py")
    checklist = _write_checklist(
        tmp_path / "checklist.yaml",
        [{"id": "x", "category": "contract", "required": True, "paths": ["x.py"]}],
    )
    report = preflight.build_report(checklist, repo)
    assert "presence-only" in report["claim_boundary"]
    assert "readiness" in report["claim_boundary"]


@pytest.mark.parametrize(
    "bad_payload",
    [
        "not a mapping\n",
        "schema_version: wrong\nprerequisites: []\n",
        f"schema_version: {SCHEMA}\nprerequisites: []\n",
    ],
)
def test_invalid_checklist_raises(tmp_path: Path, bad_payload: str) -> None:
    """Malformed checklists raise ValueError (surfaced as exit code 2 in main)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    checklist = tmp_path / "bad.yaml"
    checklist.write_text(bad_payload, encoding="utf-8")
    with pytest.raises(ValueError):
        preflight.build_report(checklist, repo)
    assert preflight.main(["--checklist", str(checklist), "--repo-root", str(repo)]) == 2


def test_entry_without_paths_raises(tmp_path: Path) -> None:
    """A prerequisite entry without paths is rejected."""
    repo = tmp_path / "repo"
    repo.mkdir()
    checklist = _write_checklist(
        tmp_path / "checklist.yaml",
        [{"id": "nopaths", "category": "contract", "required": True}],
    )
    with pytest.raises(ValueError):
        preflight.build_report(checklist, repo)


def test_json_output_is_valid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The --json flag emits parseable JSON with the schema version."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _touch(repo, "x.py")
    checklist = _write_checklist(
        tmp_path / "checklist.yaml",
        [{"id": "x", "category": "contract", "required": True, "paths": ["x.py"]}],
    )
    preflight.main(["--checklist", str(checklist), "--repo-root", str(repo), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == SCHEMA
    assert payload["prerequisites_satisfied"] is True


def test_repo_checklist_resolves_against_real_tree() -> None:
    """The committed v0.1 checklist references only paths that exist on this branch.

    This keeps the checklist honest: every declared prerequisite owner must be a
    real canonical path so the preflight cannot silently drift from the repo.
    """
    repo_root = preflight.get_repository_root()
    checklist = repo_root / preflight.DEFAULT_CHECKLIST_PATH
    report = preflight.build_report(checklist, repo_root)
    assert report["required_missing"] == [], report["required_missing"]
    assert report["prerequisites_satisfied"] is True
