"""Fail-closed tests for the future Chapter 7 v2 admission boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scripts.analysis import build_ch7_evidence_package_v2 as builder
from scripts.analysis import verify_ch7_evidence_admission_v2 as verifier

SOURCE_PACKAGE = (
    Path(__file__).parents[2] / "docs/context/evidence/issue_6792_ch7_evidence_package_v1"
)


def _valid_receipt() -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema_version": "ch7-evidence-admission.v2",
        "issue": 7087,
        "status": "admitted",
        "package": {"sha256sums_sha256": digest, "manifest_sha256": digest},
        "source": {
            "v1_package_sha256sums": digest,
            "v1_manifest_sha256": digest,
            "v1_audit_member_sha256": digest,
            "v1_reduced_atlas_member_sha256": digest,
            "portfolio_config_sha256": digest,
            "source_registry_sha256": digest,
        },
        "approval": {
            "approval_id": "issue7087-comment-123456",
            "approval_url": "https://github.com/ll7/robot_sf_ll7/issues/7087#issuecomment-123456",
            "decision": "approve",
        },
        "scope": {
            "claim_boundary": "release-cell descriptive v2 projection",
            "forbidden_claims": [
                "matched_comparison",
                "causal_divergence",
                "counterfactual_branching",
                "trajectory_divergence",
                "collision_metric_semantics",
            ],
        },
        "roles": {
            "available": {
                "cross_topology_inversion": {"grain": "release_cell"},
                "cross_mechanism_inversion": {"grain": "release_cell"},
                "feasibility_criticism": {"grain": "release_cell_geometry"},
            }
        },
        "retrieval": {
            "source_package_key": "issue-6792/package-v1",
            "audit_member_key": "issue-6792/package-v1/audit-campaign-atlas",
            "source_registry_key": "issue-7087/source-registry-v2",
        },
    }


def test_v2_admission_schema_is_versioned_and_strict() -> None:
    schema = json.loads(
        (
            Path(__file__).parents[2] / "robot_sf/benchmark/schemas/ch7-evidence-admission.v2.json"
        ).read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    receipt = _valid_receipt()
    assert not list(validator.iter_errors(receipt))
    mutated = json.loads(json.dumps(receipt))
    mutated["issue"] = 6792
    assert list(validator.iter_errors(mutated))


def test_blocked_builder_output_cannot_cross_the_v2_admission_boundary(tmp_path: Path) -> None:
    output = tmp_path / "package"
    builder.build_ch7_evidence_package_v2(source_package=SOURCE_PACKAGE, output=output)
    with pytest.raises(verifier.Ch7EvidenceAdmissionV2Error, match="review sidecars|admitted"):
        verifier.verify_v2_admission(output, tmp_path / "receipt.json")
