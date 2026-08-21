"""Focused tests for the fail-closed release 0.0.5 artifact surface gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.validation import check_release_surface as gate

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, payload: object) -> None:
    """Write a YAML or JSON fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _minimal_surface(tmp_path: Path, *, status: str = "blocked") -> tuple[Path, Path, Path]:
    """Create a one-entry synthetic surface for structural validator tests."""

    artifact = tmp_path / "docs" / "artifact.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact bytes\n", encoding="utf-8")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()

    manifest = tmp_path / "manifest.yaml"
    _write(
        manifest,
        {
            "schema_version": gate.CHECKSUM_MANIFEST_SCHEMA,
            "release_tag": "0.0.5",
            "entries": [{"path": "docs/artifact.txt", "sha256": artifact_sha}],
        },
    )
    # The fixture uses an explicit one-entry contract and is exercised through the
    # lower-level row/checklist helpers; the full gate remains bound to the frozen
    # 25-entry production manifest.
    disposition = tmp_path / "disposition.yaml"
    row = {
        "row_id": "artifact",
        "path": "docs/artifact.txt",
        "sha256": artifact_sha,
        "status": status,
        "reason_code": "rights_not_reviewed",
        "rationale": "synthetic rights review is intentionally pending",
        "evidence": [],
    }
    _write(
        disposition,
        {
            "schema_version": gate.DISPOSITION_SCHEMA,
            "release_id": gate.RELEASE_ID,
            "surface": gate.SURFACE,
            "approved_source_sha": gate.EXPECTED_APPROVED_SOURCE_SHA,
            "validation_base_sha": gate.EXPECTED_VALIDATION_BASE_SHA,
            "decision": {
                "issue": 7320,
                "token": "approve-artifact-only",
                "conditional": True,
                "publication_authorized": False,
            },
            "manifest": {
                "path": "configs/releases/release_0_0_5_checksum_manifest.yaml",
                "sha256": gate.EXPECTED_MANIFEST_SHA256,
                "entry_count": gate.EXPECTED_MANIFEST_ENTRY_COUNT,
            },
            "rows": [row],
        },
    )
    checklist = tmp_path / "checklist.yaml"
    _write(
        checklist,
        {
            "schema_version": gate.CHECKLIST_SCHEMA,
            "release_id": gate.RELEASE_ID,
            "surface": gate.SURFACE,
            "manifest": {
                "path": "configs/releases/release_0_0_5_checksum_manifest.yaml",
                "sha256": gate.EXPECTED_MANIFEST_SHA256,
                "entry_count": gate.EXPECTED_MANIFEST_ENTRY_COUNT,
            },
            "disposition_path": "configs/releases/release_0_0_5_surface_disposition.yaml",
            "items": [
                {
                    "item_id": "artifact",
                    "path": "docs/artifact.txt",
                    "sha256": artifact_sha,
                    "check": "rights_disposition",
                }
            ],
        },
    )
    return manifest, disposition, checklist


def test_committed_surface_is_exactly_25_paths_and_remains_blocked() -> None:
    """The shipped sidecar binds every manifest path but clears none without rights evidence."""

    report = gate.build_report(repo_root=REPO_ROOT)

    assert report["status"] == "blocked"
    assert report["manifest"]["sha256"] == gate.EXPECTED_MANIFEST_SHA256
    assert report["summary"]["manifest_paths"] == 25
    assert report["summary"]["disposition_rows"] == 25
    assert report["summary"]["counts_by_status"] == {
        "blocked": 25,
        "cleared_for": 0,
        "project_authored": 0,
    }
    assert sum(issue["code"] == "rights_blocked" for issue in report["blockers"]) == 25
    assert any(issue["code"] == "approved_source_sha_mismatch" for issue in report["blockers"])
    assert {
        issue["path"]
        for issue in report["blockers"]
        if issue["code"] == "manifest_checksum_mismatch"
    } == {
        "docs/context/evidence/issue_5034_control_action_latency_sweep/README.md",
        "docs/context/evidence/issue_5034_control_action_latency_sweep/manifest.sha256",
        "configs/benchmarks/issue_5592_cross_matrix_preregistration.yaml",
    }


def test_manifest_path_inventory_is_the_authoritative_25_entry_set() -> None:
    """The exact path list is mechanically sourced from the checksum manifest."""

    payload = yaml.safe_load((REPO_ROOT / gate.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
    paths = [entry["path"] for entry in payload["entries"]]
    report = gate.build_report(repo_root=REPO_ROOT)

    assert report["manifest"]["paths"] == paths
    assert len(paths) == len(set(paths)) == 25


def test_clear_status_requires_explicit_evidence_record(tmp_path: Path) -> None:
    """A project-authored status without a bound evidence record fails closed."""

    manifest, disposition, checklist = _minimal_surface(tmp_path, status="project_authored")
    payload = yaml.safe_load(disposition.read_text(encoding="utf-8"))
    payload["rows"][0]["evidence"] = []
    _write(disposition, payload)

    issues: list[dict] = []
    manifest_payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    entries = gate._manifest_entries(
        manifest_payload,
        repo_root=tmp_path,
        issues=issues,
        enforce_frozen_shape=False,
    )
    gate._validate_disposition_rows(
        payload,
        entries=entries,
        repo_root=tmp_path,
        issues=issues,
    )

    assert any(issue["code"] == "clear_without_evidence" for issue in issues)
    assert checklist.exists()


def test_explicit_project_authorship_record_can_clear_synthetic_row(tmp_path: Path) -> None:
    """A separate exact-path, exact-digest authorship record is mechanically acceptable."""

    manifest, disposition, _ = _minimal_surface(tmp_path, status="project_authored")
    artifact_sha = hashlib.sha256((tmp_path / "docs/artifact.txt").read_bytes()).hexdigest()
    rights = tmp_path / "docs" / "rights_record.yaml"
    rights.write_text(
        "\n".join(
            [
                "evidence_type: project_authorship",
                "status: project_authored",
                "surface: artifact_only_zenodo",
                "target_path: docs/artifact.txt",
                f"target_sha256: {artifact_sha}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rights_sha = hashlib.sha256(rights.read_bytes()).hexdigest()
    payload = yaml.safe_load(disposition.read_text(encoding="utf-8"))
    payload["rows"][0]["evidence"] = [
        {
            "evidence_type": "project_authorship",
            "path": "docs/rights_record.yaml",
            "sha256": rights_sha,
            "target_path": "docs/artifact.txt",
            "target_sha256": artifact_sha,
            "rights_holder": "Robot SF project authors",
            "basis": "explicit project-authorship record",
            "review_reference": "synthetic-maintainer-review",
        }
    ]
    _write(disposition, payload)
    issues: list[dict] = []
    entries = gate._manifest_entries(
        yaml.safe_load(manifest.read_text(encoding="utf-8")),
        repo_root=tmp_path,
        issues=issues,
        enforce_frozen_shape=False,
    )
    counts = gate._validate_disposition_rows(
        payload,
        entries=entries,
        repo_root=tmp_path,
        issues=issues,
    )

    assert issues == []
    assert counts == {"blocked": 0, "cleared_for": 0, "project_authored": 1}


def test_permission_required_status_is_not_a_clearance(tmp_path: Path) -> None:
    """Permission-required/unknown states cannot bypass the allowed status vocabulary."""

    manifest, disposition, _ = _minimal_surface(tmp_path, status="permission_required")
    issues: list[dict] = []
    entries = gate._manifest_entries(
        yaml.safe_load(manifest.read_text(encoding="utf-8")),
        repo_root=tmp_path,
        issues=issues,
        enforce_frozen_shape=False,
    )
    gate._validate_disposition_rows(
        yaml.safe_load(disposition.read_text(encoding="utf-8")),
        entries=entries,
        repo_root=tmp_path,
        issues=issues,
    )

    assert any(issue["code"] == "disposition_status" for issue in issues)


def test_noncanonical_sidecar_cli_overrides_are_rejected(tmp_path: Path) -> None:
    """Production validation cannot replace committed disposition/checklist controls."""

    with pytest.raises(SystemExit):
        gate._parse_args(["--disposition", str(tmp_path / "disposition.yaml")])


def test_non_checkout_source_root_is_blocked(tmp_path: Path) -> None:
    """A copied byte tree cannot impersonate the exact approved Git checkout."""

    sha, error = gate._checkout_sha(tmp_path)

    assert sha is None
    assert error is not None


def test_authorization_fields_cannot_be_weakened() -> None:
    """The sidecar cannot widen the maintainer's conditional route selection."""

    payload = yaml.safe_load((REPO_ROOT / gate.DEFAULT_DISPOSITION).read_text(encoding="utf-8"))
    payload["decision"]["issue"] = 9999
    payload["decision"]["conditional"] = False
    payload["decision"]["publication_authorized"] = True
    issues: list[dict] = []

    gate._validate_disposition_header(payload, issues=issues)

    assert {issue["code"] for issue in issues} == {
        "decision_conditional",
        "decision_issue",
        "decision_publication_authorized",
    }
