"""Contract tests for the Chapter 7 v2 build-provenance receipt."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.analysis import ch7_evidence_build_receipt as receipt

REPOSITORY = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def built_receipt(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build one real receipt so mutation tests exercise the live package contract."""

    output = tmp_path_factory.mktemp("ch7-receipt") / "receipt.json"
    return receipt.build_receipt(repository=REPOSITORY, output=output, source_commit="HEAD")


def _write_mutated_receipt(tmp_path: Path, payload: dict[str, object], *, mutate: callable) -> Path:
    mutated = copy.deepcopy(payload)
    mutate(mutated)
    integrity = mutated["integrity"]
    assert isinstance(integrity, dict)
    integrity["receipt_payload_sha256"] = "0" * 64
    integrity["receipt_payload_sha256"] = receipt._receipt_payload_hash(mutated)
    path = tmp_path / "mutated-receipt.json"
    path.write_bytes(receipt.canonical_bytes(mutated))
    return path


def test_receipt_builds_two_matching_outputs_and_verifies(tmp_path: Path) -> None:
    """The receipt proves two independent builds match the durable package without admission."""

    path = tmp_path / "receipt.json"
    payload = receipt.build_receipt(repository=REPOSITORY, output=path, source_commit="HEAD")
    assert payload["status"] == "verified_build_provenance"
    assert payload["admission_boundary"] == {
        "admission_receipt_created": False,
        "domain_approval_recorded": False,
        "publication_authorized": False,
        "benchmark_result_admitted": False,
    }
    assert payload["rebuilds"]["byte_identical"] is True
    result = receipt.verify_receipt(repository=REPOSITORY, receipt_path=path)
    assert result["status"] == "verified"
    assert result["admission_status"] == "not_admitted"
    assert result["rebuilds_verified"] is True


def test_receipt_integrity_hash_excludes_only_itself() -> None:
    """Changing the stored payload hash alone cannot create a circular self-hash."""

    payload = {
        "value": "stable",
        "integrity": {
            "canonicalization": receipt.CANONICALIZATION,
            "scope": receipt.INTEGRITY_SCOPE,
            "receipt_payload_sha256": "0" * 64,
        },
    }
    first = receipt._receipt_payload_hash(payload)
    payload["integrity"]["receipt_payload_sha256"] = "f" * 64
    assert receipt._receipt_payload_hash(payload) == first


def test_changed_source_hash_is_rejected(built_receipt: dict[str, object], tmp_path: Path) -> None:
    """A changed builder source binding cannot reuse the old receipt."""

    def mutate(payload: dict[str, object]) -> None:
        payload["source"]["builder"]["sha256"] = "0" * 64

    path = _write_mutated_receipt(tmp_path, built_receipt, mutate=mutate)
    with pytest.raises(receipt.Ch7EvidenceBuildReceiptError, match="pinned source hash"):
        receipt.verify_receipt(repository=REPOSITORY, receipt_path=path, rebuild=False)


def test_changed_dependency_identity_is_rejected(
    built_receipt: dict[str, object], tmp_path: Path
) -> None:
    """A changed resolved-dependency digest cannot pass the environment contract."""

    def mutate(payload: dict[str, object]) -> None:
        payload["environment"]["resolved_dependencies"]["sha256"] = "0" * 64

    path = _write_mutated_receipt(tmp_path, built_receipt, mutate=mutate)
    with pytest.raises(receipt.Ch7EvidenceBuildReceiptError, match="dependency identity"):
        receipt.verify_receipt(repository=REPOSITORY, receipt_path=path, rebuild=False)


def test_changed_output_tree_is_rejected(built_receipt: dict[str, object], tmp_path: Path) -> None:
    """A changed durable package tree cannot be mistaken for the recorded output."""

    def mutate(payload: dict[str, object]) -> None:
        payload["package"]["tree_sha256"] = "0" * 64

    path = _write_mutated_receipt(tmp_path, built_receipt, mutate=mutate)
    with pytest.raises(receipt.Ch7EvidenceBuildReceiptError, match="tracked package hash"):
        receipt.verify_receipt(repository=REPOSITORY, receipt_path=path, rebuild=False)


def test_schema_is_strict_and_receipt_is_canonical(built_receipt: dict[str, object]) -> None:
    """The durable JSON shape rejects unknown fields and is written canonically."""

    schema = json.loads(
        (REPOSITORY / "robot_sf/benchmark/schemas/ch7-evidence-build-receipt.v1.json").read_text(
            encoding="utf-8"
        )
    )
    from jsonschema import Draft202012Validator

    assert not list(Draft202012Validator(schema).iter_errors(built_receipt))
    mutated = copy.deepcopy(built_receipt)
    mutated["unexpected"] = True
    assert list(Draft202012Validator(schema).iter_errors(mutated))
