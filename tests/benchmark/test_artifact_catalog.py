"""Tests for the ``artifact_catalog.v1`` contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.artifact_catalog import (
    ARTIFACT_CATALOG_SCHEMA_VERSION,
    ArtifactCatalogValidationError,
    artifact_catalog_from_dict,
    load_artifact_catalog,
    main,
    sha256_file,
    validate_artifact_catalog,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "artifact_catalog" / "v1"
VALID_CATALOG = FIXTURE_DIR / "valid_catalog.yaml"


def _payload() -> dict[str, object]:
    """Return a mutable valid fixture catalog payload."""

    payload = yaml.safe_load(VALID_CATALOG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_load_valid_catalog_as_typed_metadata() -> None:
    """The fixture catalog should expose stable artifact IDs and file checksums."""

    catalog = load_artifact_catalog(VALID_CATALOG)

    assert catalog.schema_version == ARTIFACT_CATALOG_SCHEMA_VERSION
    assert catalog.catalog_id == "fixture_camera_ready_artifacts"
    assert [artifact.artifact_id for artifact in catalog.artifacts] == [
        "fig_benchmark_outcome_matrix",
        "tab_planner_execution_modes",
    ]
    assert catalog.artifacts[0].outputs["png"].path == "fig_benchmark_outcome_matrix.png"


@pytest.mark.parametrize(
    ("mutate", "expected_path", "expected_fragment"),
    [
        (
            lambda payload: payload["artifacts"].append(deepcopy(payload["artifacts"][0])),
            "/artifacts/2/artifact_id",
            "duplicate artifact_id 'fig_benchmark_outcome_matrix'",
        ),
        (
            lambda payload: payload["artifacts"][0]["source_files"][0].__setitem__(
                "path",
                "missing_summary.json",
            ),
            "/artifacts/0/source_files/0/path",
            "path does not exist: missing_summary.json",
        ),
        (
            lambda payload: payload["artifacts"][0]["outputs"]["png"].__setitem__(
                "path",
                "missing_output.png",
            ),
            "/artifacts/0/outputs/png/path",
            "path does not exist: missing_output.png",
        ),
        (
            lambda payload: payload["artifacts"][0]["outputs"]["png"].__setitem__(
                "path",
                "output/generated.png",
            ),
            "/artifacts/0/outputs/png/path",
            "local-only artifact reference is not durable",
        ),
        (
            lambda payload: payload["artifacts"][0]["outputs"]["png"].__setitem__(
                "sha256",
                "0" * 64,
            ),
            "/artifacts/0/outputs/png/sha256",
            "checksum mismatch",
        ),
    ],
)
def test_catalog_validation_reports_actionable_issues(
    mutate,
    expected_path: str,
    expected_fragment: str,
) -> None:
    """Catalog failures should identify the exact artifact field to fix."""

    payload = _payload()
    mutate(payload)

    issues = artifact_catalog_from_dict_invalid(payload)

    assert any(
        issue.path == expected_path and expected_fragment in issue.message for issue in issues
    )


def artifact_catalog_from_dict_invalid(payload: dict[str, object]):
    """Return validation issues from the typed loader failure path."""

    with pytest.raises(ArtifactCatalogValidationError) as exc_info:
        artifact_catalog_from_dict(payload, catalog_path=VALID_CATALOG)
    return exc_info.value.issues


def test_cli_validation_returns_issues_for_invalid_catalog(tmp_path: Path) -> None:
    """The path-based validator should be usable by scripts and CI."""

    payload = _payload()
    payload["artifacts"][1]["outputs"]["md"]["path"] = "missing_table.md"
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    issues = validate_artifact_catalog(catalog_path)

    assert any(issue.path == "/artifacts/1/outputs/md/path" for issue in issues)


def test_local_only_path_check_is_component_aware(tmp_path: Path) -> None:
    """Sibling names like output_report.md should not trip the output/ durability guard."""

    source = tmp_path / "output_report.md"
    source.write_text("source fixture\n", encoding="utf-8")
    rendered = tmp_path / "results_table.md"
    rendered.write_text("rendered fixture\n", encoding="utf-8")
    catalog = {
        "schema_version": ARTIFACT_CATALOG_SCHEMA_VERSION,
        "catalog_id": "path_boundary_fixture",
        "artifacts": [
            {
                "artifact_id": "tab_path_boundary",
                "artifact_kind": "table",
                "source_kind": "benchmark_campaign",
                "source_files": [{"path": source.name, "sha256": sha256_file(source)}],
                "outputs": {"md": {"path": rendered.name, "sha256": sha256_file(rendered)}},
                "generation_command": "fixture",
                "generation_commit": "44f4f364",
                "claim_boundary": "fixture only",
            }
        ],
    }
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

    assert validate_artifact_catalog(catalog_path) == []


def test_cli_main_fails_on_duplicate_artifact_ids(tmp_path: Path, capsys) -> None:
    """The command-line entry point should fail closed for duplicate semantic IDs."""

    payload = _payload()
    payload["artifacts"].append(deepcopy(payload["artifacts"][0]))
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    exit_code = main([str(catalog_path), "--json"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert '"ok": false' in captured.out
    assert "duplicate artifact_id" in captured.out
