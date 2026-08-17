"""Tests for the fail-closed Chapter 7 v2 build provenance receipt."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.analysis import verify_ch7_evidence_build_receipt_v1 as receipt

REPO_ROOT = Path(__file__).parents[2]
RECEIPT_PATH = (
    REPO_ROOT / "docs/context/evidence/issue_7322_ch7_evidence_build_receipt.v1.json"
)


def _write_mutated_receipt(tmp_path: Path, mutate) -> Path:
    wrapper = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    payload = copy.deepcopy(wrapper["payload"])
    mutate(payload)
    mutated = receipt._receipt_wrapper(payload)
    path = tmp_path / "receipt.json"
    path.write_bytes(receipt._canonical_bytes(mutated))
    return path


def test_durable_receipt_verifies_and_preserves_non_admission_boundary() -> None:
    result = receipt.verify_receipt(RECEIPT_PATH, repo_root=REPO_ROOT)

    assert result["status"] == "build_provenance_verified"
    assert result["admission_status"] == "not_admitted"
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))["payload"]
    assert payload["determinism"]["outputs_match"] is True
    assert payload["admission_boundary"]["paper_facing_use_authorized"] is False


def test_receipt_payload_hash_is_non_circular() -> None:
    wrapper = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    assert wrapper["receipt_hash"]["hashed_object"] == "payload"
    assert wrapper["receipt_hash"]["sha256"] == receipt._sha256_bytes(
        receipt._canonical_bytes(wrapper["payload"])
    )


def test_changed_builder_source_is_rejected(tmp_path: Path) -> None:
    path = _write_mutated_receipt(
        tmp_path,
        lambda payload: payload["implementation"]["builder"].update(
            {"sha256": "0" * 64}
        ),
    )

    with pytest.raises(
        receipt.Ch7EvidenceBuildReceiptError, match="implementation hash mismatch"
    ):
        receipt.verify_receipt(path, repo_root=REPO_ROOT)


def test_changed_config_is_rejected(tmp_path: Path) -> None:
    path = _write_mutated_receipt(
        tmp_path,
        lambda payload: payload["inputs"]["v2_config"].update({"sha256": "0" * 64}),
    )

    with pytest.raises(
        receipt.Ch7EvidenceBuildReceiptError, match="input hash mismatch"
    ):
        receipt.verify_receipt(path, repo_root=REPO_ROOT)


def test_changed_dependency_identity_is_rejected(tmp_path: Path) -> None:
    path = _write_mutated_receipt(
        tmp_path,
        lambda payload: payload["environment"]["project"]["lock"].update(
            {"sha256": "0" * 64}
        ),
    )

    with pytest.raises(
        receipt.Ch7EvidenceBuildReceiptError, match="dependency identity changed"
    ):
        receipt.verify_receipt(path, repo_root=REPO_ROOT)


def test_changed_output_tree_is_rejected(tmp_path: Path) -> None:
    path = _write_mutated_receipt(
        tmp_path,
        lambda payload: payload["package"].update({"payload_tree_sha256": "0" * 64}),
    )

    with pytest.raises(
        receipt.Ch7EvidenceBuildReceiptError,
        match="durable package payload tree changed",
    ):
        receipt.verify_receipt(path, repo_root=REPO_ROOT)
