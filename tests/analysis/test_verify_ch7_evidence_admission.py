"""Fail-closed tests for the Chapter 7 author-admission boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.analysis import verify_ch7_evidence_admission as admission

SOURCE_SHA = "1" * 64
RELEASE_SHA = "2" * 64
COMPACT_SHA = "3" * 64
COMPACT_SUMS_SHA = "4" * 64
APPROVAL_ID = "issue6792-comment-5273889714"
APPROVAL_URL = "https://github.com/ll7/robot_sf_ll7/issues/6792#issuecomment-5273889714"
CLAIM_BOUNDARY = "Release-cell descriptive evidence only. Trace-level trajectory dossiers remain typed unavailable."


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode())


def _write_sums(root: Path) -> str:
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        entries.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="ascii")
    return _sha256(root / "SHA256SUMS")


def _write_review_sidecars(root: Path) -> None:
    for artifact in sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    ):
        relative = artifact.relative_to(root).as_posix()
        _write_json(
            Path(f"{artifact}.review.json"),
            {
                "schema_version": "evidence-review-marker.v1",
                "artifact_path": f"docs/context/evidence/{root.name}/{relative}",
                "artifact_sha256": _sha256(artifact),
                "review_marker": "AI-GENERATED NEEDS-REVIEW",
                "preserved_exact_bytes": True,
            },
        )
    _write_json(
        root / "SHA256SUMS.review.json",
        {
            "schema_version": "evidence-review-marker.v1",
            "artifact_path": f"docs/context/evidence/{root.name}/SHA256SUMS",
            "artifact_sha256": _sha256(root / "SHA256SUMS"),
            "review_marker": "AI-GENERATED NEEDS-REVIEW",
            "preserved_exact_bytes": True,
        },
    )


def _make_external_inputs(tmp_path: Path) -> dict[str, Path | str]:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "source.dat").write_bytes(b"approved-source")
    source_sha = _write_sums(source)
    _write_json(source / "package_complete.json", {"sha256sums_sha256": source_sha})
    source_complete_sha = _sha256(source / "package_complete.json")

    release = tmp_path / "release.tar.gz"
    release.write_bytes(b"approved-release")
    release_sha = _sha256(release)

    compact = tmp_path / "compact"
    compact.mkdir()
    (compact / "compact_packet.json").write_bytes(b'{"issue":6814}\n')
    compact_sums_sha = _write_sums(compact)
    compact_sha = _sha256(compact / "compact_packet.json")

    return {
        "source": source,
        "source_sha": source_sha,
        "source_complete_sha": source_complete_sha,
        "release": release,
        "release_sha": release_sha,
        "compact": compact,
        "compact_sha": compact_sha,
        "compact_sums_sha": compact_sums_sha,
    }


def _make_package(tmp_path: Path, inputs: dict[str, Path | str]) -> Path:
    package = tmp_path / "package"
    manifest = {
        "schema_version": "ch7-evidence-package.v1",
        "issue": 6792,
        "status": "blocked_pending_domain_approval",
        "admission_status": "not_admitted",
        "source_integrity_gate": "blocked_pending_domain_approval",
        "source": {
            "approved_package_sha256sums": inputs["source_sha"],
            "release_archive_sha256": inputs["release_sha"],
            "issue6814_compact_packet_sha256": inputs["compact_sha"],
        },
        "inputs": {
            "portfolio_config": {"name": "fixture", "sha256": "5" * 64},
            "source_package_member_count": 1,
        },
        "counts": {"requested": 90, "admitted": 88, "excluded": 2},
        "atlas": {"audit_cells": 672, "publication_cells": 20, "planner_arms": 14},
        "roles": {
            "cross_cell_inversion": {"status": "available", "grain": "release_cell"},
            "feasibility_criticism": {
                "status": "available",
                "grain": "release_cell_geometry",
            },
            "planner_upset": {
                "status": "unavailable",
                "reason": "#6814 incompatible seed-118 starts and no shared-start receipt",
            },
            "seed_sensitivity": {
                "status": "unavailable",
                "reason": "#6814 shared_prefix=false and unequal starts",
            },
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "raw_traces_included": False,
        "release_archive_included": False,
        "deterministic_serialization": "strict-json-sort-keys-utf8-newline.v1",
    }
    _write_json(package / "manifest.json", manifest)
    _write_json(
        package / "audit" / "summary.json", {"requested": 90, "admitted": 88, "excluded": 2}
    )
    _write_sums(package)
    _write_review_sidecars(package)
    return package


def _make_receipt(tmp_path: Path, package: Path, inputs: dict[str, Path | str]) -> Path:
    registry = tmp_path / "source-registry.json"
    registry_payload = {
        "schema_version": "case-source-integrity-registry.v1",
        "approved_sources": [
            {
                "approval_id": APPROVAL_ID,
                "approval_url": APPROVAL_URL,
                "status": "approved",
                "source_package_sha256sums": inputs["source_sha"],
                "source_package_complete_sha256": inputs["source_complete_sha"],
                "release_archive_sha256": inputs["release_sha"],
                "compact_packet_sha256": inputs["compact_sha"],
                "compact_sha256sums_sha256": inputs["compact_sums_sha"],
                "source_package_key": "issue-6792/source-package-6412",
                "release_archive_key": "issue-6792/release-0.0.3",
                "compact_packet_key": "issue-6792/compact-6814",
            }
        ],
    }
    _write_json(registry, registry_payload)
    receipt = tmp_path / "admission.json"
    _write_json(
        receipt,
        {
            "schema_version": "ch7-evidence-admission.v1",
            "issue": 6792,
            "status": "admitted",
            "package": {
                "sha256sums_sha256": _sha256(package / "SHA256SUMS"),
                "manifest_sha256": _sha256(package / "manifest.json"),
            },
            "source": {
                "source_package_sha256sums": inputs["source_sha"],
                "source_package_complete_sha256": inputs["source_complete_sha"],
                "release_archive_sha256": inputs["release_sha"],
                "compact_packet_sha256": inputs["compact_sha"],
                "compact_sha256sums_sha256": inputs["compact_sums_sha"],
                "source_registry_sha256": _sha256(registry),
            },
            "approval": {
                "approval_id": APPROVAL_ID,
                "approval_url": APPROVAL_URL,
                "decision": "approve",
            },
            "scope": {
                "claim_boundary": CLAIM_BOUNDARY,
                "forbidden_claims": list(admission.FORBIDDEN_CLAIMS),
            },
            "roles": {
                "available": {
                    "cross_cell_inversion": {"grain": "release_cell"},
                    "feasibility_criticism": {"grain": "release_cell_geometry"},
                },
                "unavailable": {
                    "planner_upset": {
                        "grain": "trace",
                        "reasons": [
                            "#6814 incompatible seed-118 starts and no shared-start receipt"
                        ],
                    },
                    "seed_sensitivity": {
                        "grain": "trace",
                        "reasons": ["#6814 shared_prefix=false and unequal starts"],
                    },
                },
            },
            "retrieval": {
                "source_package_key": "issue-6792/source-package-6412",
                "release_archive_key": "issue-6792/release-0.0.3",
                "compact_packet_key": "issue-6792/compact-6814",
            },
        },
    )
    return receipt


def _verify_args(tmp_path: Path) -> dict[str, Path]:
    inputs = _make_external_inputs(tmp_path)
    package = _make_package(tmp_path, inputs)
    receipt = _make_receipt(tmp_path, package, inputs)
    return {
        "package": package,
        "registry": tmp_path / "source-registry.json",
        "receipt": receipt,
        "source": inputs["source"],
        "release": inputs["release"],
        "compact": inputs["compact"],
    }


def test_valid_admission_receipt_binds_all_inputs(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    result = admission.verify_admission(
        package_dir=args["package"],
        source_registry=args["registry"],
        receipt=args["receipt"],
        source_package=args["source"],
        release_archive=args["release"],
        compact_dir=args["compact"],
    )
    assert result["status"] == "admitted"
    assert result["source"]["compact_packet_sha256"] == _sha256(
        args["compact"] / "compact_packet.json"
    )
    assert result["scope"]["forbidden_claims"] == list(admission.FORBIDDEN_CLAIMS)
    assert result["roles"]["unavailable"]["seed_sensitivity"]["grain"] == "trace"
    assert result["approval_url"] == APPROVAL_URL


def test_unbound_review_sidecar_fails_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    _write_json(
        args["package"] / "unbound.json.review.json",
        {
            "schema_version": "evidence-review-marker.v1",
            "artifact_path": "docs/context/evidence/fixture/unbound.json",
            "artifact_sha256": "f" * 64,
            "review_marker": "AI-GENERATED NEEDS-REVIEW",
            "preserved_exact_bytes": True,
        },
    )
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="unbound review sidecars"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_missing_review_sidecar_fails_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    (args["package"] / "manifest.json.review.json").unlink()
    with pytest.raises(
        admission.Ch7EvidenceAdmissionError, match="missing required review sidecars"
    ):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_review_sidecar_artifact_path_must_be_exact(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    sidecar = args["package"] / "manifest.json.review.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["artifact_path"] = "docs/context/evidence/package/wrong/manifest.json"
    _write_json(sidecar, payload)
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="wrong artifact"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_review_sidecar_artifact_mutation_fails_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    sidecar = args["package"] / "manifest.json.review.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["artifact_sha256"] = "f" * 64
    _write_json(sidecar, payload)
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="review sidecar artifact hash"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_source_member_mutation_fails_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    (args["source"] / "source.dat").write_bytes(b"tampered")
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="source package member hash"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_nested_checksum_file_fails_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    nested = args["package"] / "audit" / "SHA256SUMS"
    nested.write_text("", encoding="ascii")
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="unlisted"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_symlinked_package_member_fails_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    os.symlink(args["package"] / "manifest.json", args["package"] / "alias.json")
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="symlinks"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_special_package_entry_fails_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    os.mkfifo(args["package"] / "unlisted.fifo")
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="special filesystem"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_manifest_schema_mutation_fails_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    manifest = json.loads((args["package"] / "manifest.json").read_text(encoding="utf-8"))
    del manifest["inputs"]["portfolio_config"]
    _write_json(args["package"] / "manifest.json", manifest)
    manifest_sha = _sha256(args["package"] / "manifest.json")
    sums_path = args["package"] / "SHA256SUMS"
    sums = [
        f"{manifest_sha}  manifest.json" if line.endswith("  manifest.json") else line
        for line in sums_path.read_text(encoding="ascii").splitlines()
    ]
    sums_path.write_text("\n".join(sums) + "\n", encoding="ascii")
    sidecar = args["package"] / "manifest.json.review.json"
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["artifact_sha256"] = manifest_sha
    _write_json(sidecar, sidecar_payload)
    sums_sha = _sha256(sums_path)
    sums_sidecar = args["package"] / "SHA256SUMS.review.json"
    sums_sidecar_payload = json.loads(sums_sidecar.read_text(encoding="utf-8"))
    sums_sidecar_payload["artifact_sha256"] = sums_sha
    _write_json(sums_sidecar, sums_sidecar_payload)
    receipt = json.loads(args["receipt"].read_text(encoding="utf-8"))
    receipt["package"]["manifest_sha256"] = manifest_sha
    receipt["package"]["sha256sums_sha256"] = sums_sha
    _write_json(args["receipt"], receipt)
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="package manifest schema"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [("package", "manifest_sha256"), ("source", "release_archive_sha256")],
)
def test_digest_mutation_fails_closed(tmp_path: Path, section: str, field: str) -> None:
    args = _verify_args(tmp_path)
    payload = json.loads(args["receipt"].read_text(encoding="utf-8"))
    payload[section][field] = "f" * 64
    _write_json(args["receipt"], payload)
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="digest"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_forged_approval_or_registry_entry_fails_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    payload = json.loads(args["receipt"].read_text(encoding="utf-8"))
    payload["approval"]["approval_id"] = "issue6792-comment-9999999999"
    _write_json(args["receipt"], payload)
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="registry|schema"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_approval_whitespace_fails_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    payload = json.loads(args["receipt"].read_text(encoding="utf-8"))
    payload["approval"]["approval_url"] += "\n"
    _write_json(args["receipt"], payload)
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="approval URL|schema"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_role_scope_and_forbidden_claim_mutation_fail_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    payload = json.loads(args["receipt"].read_text(encoding="utf-8"))
    payload["roles"]["available"]["cross_cell_inversion"]["grain"] = "trace"
    payload["scope"]["forbidden_claims"].remove("matched_comparison")
    _write_json(args["receipt"], payload)
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="schema"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_package_extra_file_and_compact_extra_file_fail_closed(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    (args["package"] / "unlisted.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="unlisted"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )

    args = _verify_args(tmp_path / "compact-case")
    (args["compact"] / "unlisted.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(admission.Ch7EvidenceAdmissionError, match="unlisted"):
        admission.verify_admission(
            package_dir=args["package"],
            source_registry=args["registry"],
            receipt=args["receipt"],
            source_package=args["source"],
            release_archive=args["release"],
            compact_dir=args["compact"],
        )


def test_cli_returns_typed_unavailable_for_invalid_receipt(tmp_path: Path) -> None:
    args = _verify_args(tmp_path)
    args["receipt"].write_text("{}\n", encoding="utf-8")
    assert (
        admission.main(
            [
                "--package-dir",
                str(args["package"]),
                "--source-registry",
                str(args["registry"]),
                "--receipt",
                str(args["receipt"]),
                "--source-package",
                str(args["source"]),
                "--release-archive",
                str(args["release"]),
                "--compact-dir",
                str(args["compact"]),
            ]
        )
        == 2
    )
