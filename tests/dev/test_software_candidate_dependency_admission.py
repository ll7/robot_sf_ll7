"""Contract tests for candidate-bound supported-dependency receipt admission."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

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
            "workflow": {"run_id": identity["workflow_run_id"]},
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
            }
        ],
        "repository_inputs": [
            {"path": SUPPORTED_DEPENDENCY_POLICY_PATH, "sha256": "c" * 64},
            {"path": SUPPORTED_DEPENDENCY_PROFILE_PATH, "sha256": "d" * 64},
        ],
        "schema_version": "robot-sf.dependency-license-inventory.v1",
        "failures": [],
        "structural_issues": [],
        "summary": {"candidate_bound": True, "status": "complete", "unresolved_count": 0},
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
            "unresolved",
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
        )
