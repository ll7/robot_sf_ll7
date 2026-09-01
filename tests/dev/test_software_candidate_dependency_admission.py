"""Contract tests for candidate-bound supported-dependency receipt admission."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

import scripts.dev.software_candidate_manifest as candidate_manifest
from scripts.dev.software_candidate_manifest import (
    SUPPORTED_DEPENDENCY_EXTRA_IDS,
    SUPPORTED_DEPENDENCY_POLICY_PATH,
    SUPPORTED_DEPENDENCY_POLICY_SCHEMA_VERSION,
    SUPPORTED_DEPENDENCY_PROFILE_IDS,
    SUPPORTED_DEPENDENCY_PROFILE_PATH,
    SUPPORTED_DEPENDENCY_PROFILE_SCHEMA_VERSION,
    SUPPORTED_DEPENDENCY_REPORT_NAME,
    CandidateError,
    _validate_supported_dependency_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def _identity() -> dict[str, object]:
    members = [
        {
            "filename": "robot_sf-0.0.6-py3-none-any.whl",
            "kind": "wheel",
            "sha256": "1" * 64,
            "size": 10,
        },
        {
            "filename": "robot_sf-0.0.6.tar.gz",
            "kind": "sdist",
            "sha256": "2" * 64,
            "size": 11,
        },
        {
            "filename": "robot_sf-0.0.6.cyclonedx.json",
            "kind": "sbom",
            "sha256": "3" * 64,
            "size": 12,
        },
        {
            "filename": "candidate-provenance.json",
            "kind": "provenance",
            "sha256": "4" * 64,
            "size": 13,
        },
    ]
    return {
        "manifest_sha256": "a" * 64,
        "members": members,
        "package": {"name": "robot_sf", "version": "0.0.6"},
        "sbom_sha256": "3" * 64,
        "source_sha": "b" * 40,
        "workflow_run_id": "123456",
        "workflow_run_attempt": 1,
        "materialization": {
            "candidate_commit_sha": "c" * 40,
            "candidate_tree_sha": "d" * 40,
            "policy_path": "scripts/validation/software_candidate_policy.v1.json",
            "policy_sha256": "e" * 64,
            "source_inventory_path": "scripts/validation/asset_rights_inventory.v1.yaml",
            "source_inventory_sha256": "f" * 64,
            "candidate_inventory_path": "scripts/validation/software_candidate_asset_rights.v1.json",
            "candidate_metadata_path": "SOFTWARE_CANDIDATE.json",
        },
    }


def _report(identity: dict[str, object]) -> dict[str, object]:
    return {
        "candidate_binding": {
            "manifest_sha256": identity["manifest_sha256"],
            "members": identity["members"],
            "package": identity["package"],
            "source_sha": identity["source_sha"],
            "status": "bound",
            "sbom": {"sha256": identity["sbom_sha256"]},
            "workflow": {
                "run_id": identity["workflow_run_id"],
                "run_attempt": identity["workflow_run_attempt"],
            },
            "materialization": identity["materialization"],
        },
        "policy": {
            "path": SUPPORTED_DEPENDENCY_POLICY_PATH,
            "schema_version": SUPPORTED_DEPENDENCY_POLICY_SCHEMA_VERSION,
        },
        "profile_manifest": {
            "path": SUPPORTED_DEPENDENCY_PROFILE_PATH,
            "schema_version": SUPPORTED_DEPENDENCY_PROFILE_SCHEMA_VERSION,
        },
        "profiles": [
            {
                "id": "all",
                "extras": list(SUPPORTED_DEPENDENCY_EXTRA_IDS),
                "excluded_extras": ["rllib"],
                "package_ids": ["reviewed-package@1.0.0#fixture"],
            }
        ],
        "repository_inputs": [
            {"path": SUPPORTED_DEPENDENCY_POLICY_PATH, "sha256": "c" * 64},
            {"path": SUPPORTED_DEPENDENCY_PROFILE_PATH, "sha256": "d" * 64},
        ],
        "schema_version": "robot-sf.dependency-license-inventory.v1",
        "failures": [],
        "structural_issues": [],
        "packages": [
            {
                "package_id": "reviewed-package@1.0.0#fixture",
                "normalized_name": "reviewed-package",
                "version": "1.0.0",
                "name": "reviewed-package",
                "selected_profiles": ["all"],
                "policy_disposition": "external_dependency_not_redistributed",
            }
        ],
        "summary": {
            "candidate_bound": True,
            "policy_pending_package_count": 0,
            "selected_package_count": 1,
            "status": "complete",
            "structural_issue_count": 0,
            "unresolved_count": 0,
        },
        "surface": {"profile_ids": list(SUPPORTED_DEPENDENCY_PROFILE_IDS)},
    }


def test_supported_dependency_report_binds_exact_candidate_and_input_digests(
    tmp_path: Path,
) -> None:
    identity = _identity()
    report_path = tmp_path / SUPPORTED_DEPENDENCY_REPORT_NAME
    report_path.write_text(
        json.dumps(_report(identity), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    binding = _validate_supported_dependency_report(
        report_path,
        identity=identity,
        source_sha=identity["source_sha"],
        tree_sha256="e" * 64,
        workflow_run_attempt=identity["workflow_run_attempt"],
        materialization=identity["materialization"],
    )

    assert binding["candidate_manifest_sha256"] == identity["manifest_sha256"]
    assert binding["candidate_tree_sha256"] == "e" * 64
    assert binding["policy_sha256"] == "c" * 64
    assert binding["profile_manifest_sha256"] == "d" * 64
    assert binding["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert binding["unresolved_count"] == 0


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            "summary",
            {"candidate_bound": True, "status": "complete", "unresolved_count": 1},
            "summary|unresolved",
        ),
        ("candidate_binding", {"status": "bound"}, "source SHA"),
        ("surface", {"profile_ids": []}, "profile surface"),
    ],
)
def test_supported_dependency_report_rejects_unbound_or_unresolved_variants(
    tmp_path: Path,
    path: str,
    value: object,
    message: str,
) -> None:
    identity = _identity()
    payload = _report(identity)
    payload[path] = value
    report_path = tmp_path / SUPPORTED_DEPENDENCY_REPORT_NAME
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CandidateError, match=message):
        _validate_supported_dependency_report(
            report_path,
            identity=identity,
            source_sha=identity["source_sha"],
            tree_sha256="e" * 64,
            workflow_run_attempt=identity["workflow_run_attempt"],
            materialization=identity["materialization"],
        )


def test_supported_dependency_report_rejects_forged_complete_summary(tmp_path: Path) -> None:
    """A complete/zero summary cannot hide a pending selected package row."""
    identity = _identity()
    payload = _report(identity)
    payload["packages"] = [
        {
            "name": "unreviewed-package",
            "selected_profiles": ["all"],
            "policy_disposition": "review_required",
        }
    ]
    report_path = tmp_path / SUPPORTED_DEPENDENCY_REPORT_NAME
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CandidateError, match="summary.policy_pending_package_count"):
        _validate_supported_dependency_report(
            report_path,
            identity=identity,
            source_sha=identity["source_sha"],
            tree_sha256="e" * 64,
            workflow_run_attempt=identity["workflow_run_attempt"],
            materialization=identity["materialization"],
        )


def test_supported_dependency_report_rejects_hidden_pending_surface_rows(tmp_path: Path) -> None:
    """Empty selected profiles cannot hide 154 pending rows behind a zero summary."""
    identity = _identity()
    payload = _report(identity)
    rows = []
    package_ids = []
    for index in range(154):
        package_id = f"pending-{index}@1.0.0#fixture"
        package_ids.append(package_id)
        rows.append(
            {
                "package_id": package_id,
                "normalized_name": f"pending-{index}",
                "version": "1.0.0",
                "name": f"pending-{index}",
                "selected_profiles": [],
                "policy_disposition": "review_required",
            }
        )
    payload["profiles"] = [
        {
            "id": "all",
            "extras": list(SUPPORTED_DEPENDENCY_EXTRA_IDS),
            "excluded_extras": ["rllib"],
            "package_ids": package_ids,
        }
    ]
    payload["packages"] = rows
    payload["summary"] = {
        "candidate_bound": True,
        "policy_pending_package_count": 0,
        "selected_package_count": 0,
        "status": "complete",
        "structural_issue_count": 0,
        "unresolved_count": 0,
    }
    report_path = tmp_path / SUPPORTED_DEPENDENCY_REPORT_NAME
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CandidateError, match="selected-profile membership"):
        _validate_supported_dependency_report(
            report_path,
            identity=identity,
            source_sha=identity["source_sha"],
            tree_sha256="e" * 64,
            workflow_run_attempt=identity["workflow_run_attempt"],
            materialization=identity["materialization"],
        )


def test_strict_archive_gate_rejects_forbidden_rebound_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The admission helper cannot trust a producer's earlier strict-gate claim."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    wheel = bundle / "robot_sf-0.0.6-py3-none-any.whl"
    sdist = bundle / "robot_sf-0.0.6.tar.gz"
    wheel.write_bytes(b"rebound wheel with forbidden model")
    sdist.write_bytes(b"bound sdist")
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    materialization = {
        "candidate_commit_sha": "c" * 40,
        "candidate_tree_sha": "d" * 40,
        "candidate_inventory_path": "candidate-inventory.json",
    }
    manifest = {
        "materialization": materialization,
        "members": [
            {
                "filename": wheel.name,
                "kind": "wheel",
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            },
            {
                "filename": sdist.name,
                "kind": "sdist",
                "sha256": hashlib.sha256(sdist.read_bytes()).hexdigest(),
            },
        ],
        "source_sha": "b" * 40,
    }

    def reject_forbidden_archive(*_args: object, **_kwargs: object) -> None:
        raise candidate_manifest.DistributionLicenseError(
            "model artifact member is forbidden in a software distribution"
        )

    monkeypatch.setattr(candidate_manifest, "check_distribution", reject_forbidden_archive)
    with pytest.raises(CandidateError, match="strict candidate archive/tree gate failed"):
        candidate_manifest._strict_archive_gate(
            bundle, candidate_root=candidate_root, manifest=manifest
        )


def test_supported_dependency_report_rejects_wrong_filename(tmp_path: Path) -> None:
    identity = _identity()
    report_path = tmp_path / "other-report.json"
    report_path.write_text(json.dumps(_report(identity)) + "\n", encoding="utf-8")

    with pytest.raises(CandidateError, match="must be named"):
        _validate_supported_dependency_report(
            report_path,
            identity=identity,
            source_sha=identity["source_sha"],
            tree_sha256="e" * 64,
            workflow_run_attempt=identity["workflow_run_attempt"],
            materialization=identity["materialization"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [("workflow", "workflow identity"), ("materialization", "materialization identity")],
)
def test_supported_dependency_report_rejects_candidate_provenance_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """The transported report must retain the candidate attempt and materialization envelope."""
    identity = _identity()
    payload = _report(identity)
    binding = payload["candidate_binding"]
    assert isinstance(binding, dict)
    if mutation == "workflow":
        workflow = binding["workflow"]
        assert isinstance(workflow, dict)
        workflow["run_attempt"] = 2
    else:
        materialization = binding["materialization"]
        assert isinstance(materialization, dict)
        binding["materialization"] = {**materialization, "candidate_commit_sha": "a" * 40}
    report_path = tmp_path / SUPPORTED_DEPENDENCY_REPORT_NAME
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CandidateError, match=message):
        _validate_supported_dependency_report(
            report_path,
            identity=identity,
            source_sha=identity["source_sha"],
            tree_sha256="e" * 64,
            workflow_run_attempt=identity["workflow_run_attempt"],
            materialization=identity["materialization"],
        )


def test_v006_candidate_rejects_core_only_dependency_report(tmp_path: Path) -> None:
    """A core-only report cannot authorize the v0.0.6 supported extras surface."""
    identity = _identity()
    payload = _report(identity)
    payload["surface"] = {"profile_ids": ["core"]}
    report_path = tmp_path / SUPPORTED_DEPENDENCY_REPORT_NAME
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CandidateError, match="closed v0.0.6 roster"):
        _validate_supported_dependency_report(
            report_path,
            identity=identity,
            source_sha=identity["source_sha"],
            tree_sha256="e" * 64,
            workflow_run_attempt=identity["workflow_run_attempt"],
            materialization=identity["materialization"],
        )
