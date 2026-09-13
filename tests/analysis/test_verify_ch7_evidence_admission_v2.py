"""Fail-closed tests for the future Chapter 7 v2 admission boundary."""

# evidence-writer-exempt: negative tests mutate only pytest tmp_path packages to prove that a tampered atlas remains rejected after checksum regeneration; no repository evidence is written.

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scripts.analysis import build_ch7_evidence_package_v2 as builder
from scripts.analysis import verify_ch7_evidence_admission_v2 as verifier
from scripts.analysis import verify_ch7_source_registry as source_registry

SOURCE_PACKAGE = (
    Path(__file__).parents[2] / "docs/context/evidence/issue_6792_ch7_evidence_package_v1"
)
DURABLE_V2_PACKAGE = (
    Path(__file__).parents[2] / "docs/context/evidence/issue_7322_ch7_evidence_package_v2"
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
            "claim_boundary": builder.CLAIM_BOUNDARY,
            "forbidden_claims": [
                "matched_comparison",
                "causal_divergence",
                "counterfactual_branching",
                "trajectory_divergence",
                "universal_planner_ranking",
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _admitted_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    package = tmp_path / "admitted-package"
    builder.build_ch7_evidence_package_v2(source_package=SOURCE_PACKAGE, output=package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "admitted"
    manifest["admission_status"] = "admitted"
    manifest["source_integrity_gate"] = "passed"
    manifest["admission"]["status"] = "admitted"
    _write_json(package / "manifest.json", manifest)

    sums_sha = verifier._sha256_file(package / "SHA256SUMS")
    receipt = _valid_receipt()
    receipt["package"] = {
        "sha256sums_sha256": sums_sha,
        "manifest_sha256": verifier._sha256_file(package / "manifest.json"),
    }
    receipt["source"] = {
        key: manifest["source"][key]
        for key in (
            "v1_package_sha256sums",
            "v1_manifest_sha256",
            "v1_audit_member_sha256",
            "v1_reduced_atlas_member_sha256",
        )
    }
    receipt["source"].update(
        {
            "portfolio_config_sha256": manifest["inputs"]["portfolio_config"]["sha256"],
            "source_registry_sha256": manifest["source_registry"]["registry_sha256"],
        }
    )
    receipt["scope"] = {
        "claim_boundary": builder.CLAIM_BOUNDARY,
        "forbidden_claims": list(verifier.V2_FORBIDDEN_CLAIMS),
    }
    receipt["roles"] = {
        "available": {
            role: {"grain": details["grain"]} for role, details in manifest["roles"].items()
        }
    }
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, receipt)

    def _fake_verify_members(_root: Path, **_kwargs: object) -> tuple[str, list[str]]:
        return sums_sha, []

    monkeypatch.setattr(verifier.admission, "_verify_members", _fake_verify_members)
    return package, receipt_path, receipt, manifest


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


def test_check_only_accepts_fresh_and_durable_reviewed_packages(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh-package"
    builder.build_ch7_evidence_package_v2(source_package=SOURCE_PACKAGE, output=fresh)

    fresh_diagnostic = verifier.diagnose_v2_package(fresh)
    assert fresh_diagnostic["diagnostics"]["package_checksums_verified"] is True
    assert fresh_diagnostic["diagnostics"]["exclusion_boundary"] == {
        "ruling_issue": 7042,
        "status": "excluded_by_frozen_ruling",
        "metrics": list(builder.EXCLUDED_METRICS),
    }
    assert [blocker["code"] for blocker in fresh_diagnostic["diagnostics"]["blockers"]] == [
        "domain_approval_pending",
        "external_admission_receipt_required",
    ]

    durable = Path(__file__).parents[2] / "docs/context/evidence/issue_7322_ch7_evidence_package_v2"
    durable_diagnostic = verifier.diagnose_v2_package(durable)
    assert durable_diagnostic["package"] == fresh_diagnostic["package"]


def test_admission_accepts_canonical_admitted_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, receipt_path, _receipt, manifest = _admitted_fixture(tmp_path, monkeypatch)

    result = verifier.verify_v2_admission(package, receipt_path)

    assert result["status"] == "admitted"
    assert result["source_registry_sha256"] == manifest["source_registry"]["registry_sha256"]


def test_check_only_rejects_semantically_tampered_atlas_with_updated_checksum(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    builder.build_ch7_evidence_package_v2(source_package=SOURCE_PACKAGE, output=package)
    atlas_path = package / "publication/reduced_atlas.json"
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    atlas["cells"][0]["panel"] = "cross_mechanism"
    atlas_path.write_text(json.dumps(atlas, sort_keys=True, separators=(",", ":")) + "\n")
    builder._write_checksums(package)

    with pytest.raises(
        verifier.Ch7EvidenceAdmissionV2Error,
        match="cell identity differs from the portfolio",
    ):
        verifier.diagnose_v2_package(package)


def test_check_only_rejects_csv_cell_identity_drift_with_updated_checksum(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    builder.build_ch7_evidence_package_v2(source_package=SOURCE_PACKAGE, output=package)
    atlas_path = package / "publication/reduced_atlas.csv"
    with atlas_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    rows[0]["panel"] = "cross_mechanism"
    with atlas_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    builder._write_checksums(package)

    with pytest.raises(
        verifier.Ch7EvidenceAdmissionV2Error,
        match="CSV cell identity differs from the portfolio",
    ):
        verifier.diagnose_v2_package(package)


def test_blocked_builder_output_cannot_cross_the_v2_admission_boundary(tmp_path: Path) -> None:
    output = tmp_path / "package"
    builder.build_ch7_evidence_package_v2(source_package=SOURCE_PACKAGE, output=output)
    with pytest.raises(verifier.Ch7EvidenceAdmissionV2Error, match="review sidecars|admitted"):
        verifier.verify_v2_admission(output, tmp_path / "receipt.json")


def test_check_only_diagnoses_blocked_package_and_builds_template(tmp_path: Path) -> None:
    output = tmp_path / "package"
    builder.build_ch7_evidence_package_v2(source_package=SOURCE_PACKAGE, output=output)

    diagnostic = verifier.diagnose_v2_package(output)

    assert diagnostic["status"] == "blocked_pending_domain_approval"
    assert diagnostic["admission_status"] == "not_admitted"
    assert diagnostic["package"]["sha256sums_sha256"] == verifier._sha256_file(
        output / "SHA256SUMS"
    )
    assert diagnostic["diagnostics"]["package_checksums_verified"] is True
    assert diagnostic["diagnostics"]["source_registry_verified"] is True
    assert len(diagnostic["source"]["source_registry_sha256"]) == 64
    assert diagnostic["diagnostics"]["receipt_created"] is False
    assert {blocker["code"] for blocker in diagnostic["diagnostics"]["blockers"]} == {
        "domain_approval_pending",
        "external_admission_receipt_required",
    }
    assert diagnostic["diagnostics"]["exclusion_boundary"]["ruling_issue"] == 7042

    template = diagnostic["receipt_template"]
    assert template["template_status"] == "not_a_receipt"
    assert template["package"] == diagnostic["package"]
    assert (
        template["source"]["portfolio_config_sha256"]
        == diagnostic["source"]["portfolio_config_sha256"]
    )
    assert template["source"]["source_registry_sha256"] is None
    assert template["approval"]["decision"] is None
    assert template["retrieval"]["source_package_key"] is None
    assert not (output / "admission/receipt.json").exists()


def test_check_only_accepts_complete_review_sidecars_on_durable_package() -> None:
    diagnostic = verifier.diagnose_v2_package(DURABLE_V2_PACKAGE)

    assert diagnostic["status"] == "blocked_pending_domain_approval"
    assert diagnostic["admission_status"] == "not_admitted"
    assert diagnostic["diagnostics"]["package_checksums_verified"] is True
    assert diagnostic["diagnostics"]["admission_authorized"] is False
    assert diagnostic["receipt_template"]["template_status"] == "not_a_receipt"


def test_check_only_template_is_rejected_as_an_admission_receipt(tmp_path: Path) -> None:
    output = tmp_path / "package"
    builder.build_ch7_evidence_package_v2(source_package=SOURCE_PACKAGE, output=output)
    template = verifier.diagnose_v2_package(output)["receipt_template"]

    with pytest.raises(verifier.Ch7EvidenceAdmissionV2Error, match="validation failed"):
        verifier._validate(template, verifier.RECEIPT_SCHEMA, "v2 admission receipt")


def test_check_only_cli_emits_a_machine_readable_diagnostic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "package"
    builder.build_ch7_evidence_package_v2(source_package=SOURCE_PACKAGE, output=output)

    assert verifier.main(["--package-dir", str(output), "--check-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "ch7-evidence-admission-diagnostic.v1"
    assert payload["diagnostics"]["admission_authorized"] is False


def test_admission_rejects_zeroed_receipt_registry_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _package, receipt_path, receipt, _manifest = _admitted_fixture(tmp_path, monkeypatch)
    receipt["source"]["source_registry_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(
        verifier.Ch7EvidenceAdmissionV2Error,
        match="receipt source binding differs: source_registry_sha256",
    ):
        verifier.verify_v2_admission(_package, receipt_path)


def test_admission_rejects_altered_v1_source_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, receipt_path, receipt, _manifest = _admitted_fixture(tmp_path, monkeypatch)
    receipt["source"]["v1_audit_member_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(
        verifier.Ch7EvidenceAdmissionV2Error,
        match="receipt source binding differs: v1_audit_member_sha256",
    ):
        verifier.verify_v2_admission(package, receipt_path)


def test_v2_source_hashes_must_match_registry_v1_member_snapshot(tmp_path: Path) -> None:
    package = tmp_path / "package"
    builder.build_ch7_evidence_package_v2(source_package=SOURCE_PACKAGE, output=package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    registry_binding = source_registry.verify_source_registry()
    mutated = copy.deepcopy(manifest)
    mutated["source"]["v1_audit_member_sha256"] = "0" * 64

    with pytest.raises(
        verifier.Ch7EvidenceAdmissionV2Error,
        match="v2 source binding differs from the registry v1 member snapshot",
    ):
        verifier._verify_v2_source_binding(mutated, registry_binding)


def test_admission_rejects_five_of_six_forbidden_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, receipt_path, receipt, _manifest = _admitted_fixture(tmp_path, monkeypatch)
    receipt["scope"]["forbidden_claims"] = list(verifier.V2_FORBIDDEN_CLAIMS[:-1])
    _write_json(receipt_path, receipt)

    with pytest.raises(verifier.Ch7EvidenceAdmissionV2Error, match="validation failed"):
        verifier.verify_v2_admission(package, receipt_path)


def test_admission_rejects_widened_claim_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, receipt_path, receipt, _manifest = _admitted_fixture(tmp_path, monkeypatch)
    receipt["scope"]["claim_boundary"] = f"{builder.CLAIM_BOUNDARY} Also publish rankings."
    _write_json(receipt_path, receipt)

    with pytest.raises(verifier.Ch7EvidenceAdmissionV2Error, match="validation failed"):
        verifier.verify_v2_admission(package, receipt_path)


def test_admission_rejects_noncanonical_terminal_label_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, _receipt_path, _receipt, manifest = _admitted_fixture(tmp_path, monkeypatch)
    manifest["terminal_label_normalization"]["precedence"] = [
        "collision_event",
        "route_complete",
        "timeout",
        "unavailable",
    ]
    _write_json(package / "manifest.json", manifest)

    with pytest.raises(
        verifier.Ch7EvidenceAdmissionV2Error, match="terminal-label mapping changed"
    ):
        verifier.verify_v2_admission(package, _receipt_path)
