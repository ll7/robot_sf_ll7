"""Tests for the offline Zenodo release metadata validator."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from scripts.validation.check_zenodo_release_metadata import (
    DEFAULT_CITATION,
    DEFAULT_PYPROJECT,
    DEFAULT_ZENODO,
    main,
    validate_release_metadata,
)


def _repository_metadata() -> dict[str, object]:
    return json.loads(DEFAULT_ZENODO.read_text(encoding="utf-8"))


def _write_inputs(tmp_path: Path, zenodo: dict[str, object]) -> tuple[Path, Path, Path]:
    zenodo_path = tmp_path / ".zenodo.json"
    zenodo_path.write_text(json.dumps(zenodo), encoding="utf-8")
    citation_path = tmp_path / "CITATION.cff"
    citation_path.write_text(DEFAULT_CITATION.read_text(encoding="utf-8"), encoding="utf-8")
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(DEFAULT_PYPROJECT.read_text(encoding="utf-8"), encoding="utf-8")
    return zenodo_path, citation_path, pyproject_path


def test_repository_metadata_passes_offline_validation() -> None:
    assert validate_release_metadata() == []


def test_cli_passes_without_network() -> None:
    assert main([]) == 0


def test_license_drift_is_rejected(tmp_path: Path) -> None:
    metadata = _repository_metadata()
    metadata["license"] = "MIT"
    paths = _write_inputs(tmp_path, metadata)

    errors = validate_release_metadata(*paths)

    assert any("license" in error.lower() and "match" in error.lower() for error in errors)


def test_missing_authoritative_creator_is_rejected(tmp_path: Path) -> None:
    metadata = _repository_metadata()
    metadata["creators"] = [metadata["creators"][0]]
    paths = _write_inputs(tmp_path, metadata)

    errors = validate_release_metadata(*paths)

    assert any("omit authoritative" in error for error in errors)


def test_title_drift_is_rejected(tmp_path: Path) -> None:
    metadata = _repository_metadata()
    metadata["title"] = "stale-title"
    paths = _write_inputs(tmp_path, metadata)

    errors = validate_release_metadata(*paths)

    assert any("title" in error.lower() and "citation" in error.lower() for error in errors)


def test_doi_and_unknown_fields_are_rejected(tmp_path: Path) -> None:
    metadata = copy.deepcopy(_repository_metadata())
    metadata["doi"] = "10.5281/zenodo.example"
    metadata["version"] = "0.0.5"
    paths = _write_inputs(tmp_path, metadata)

    errors = validate_release_metadata(*paths)

    assert any("unsupported fields" in error for error in errors)
    assert any("must not assert a doi" in error.lower() for error in errors)


def test_tag_workflow_invokes_validator() -> None:
    workflow = Path(".github/workflows/release-functional-badge.yml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)
    assert isinstance(parsed, dict)
    assert "check_zenodo_release_metadata.py" in workflow
