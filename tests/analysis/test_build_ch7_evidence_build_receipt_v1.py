"""Fail-closed tests for the Chapter 7 v2 build receipt."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest
from jsonschema import Draft202012Validator

from scripts.analysis import build_ch7_evidence_build_receipt_v1 as receipt_tool

if TYPE_CHECKING:
    from pathlib import Path


def _generate(tmp_path: Path) -> Path:
    receipt_path = tmp_path / "build_receipt.json"
    receipt_tool.generate_receipt(
        receipt_path,
        scratch_root=tmp_path / "generate-scratch",
    )
    return receipt_path


def _write_sealed(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(receipt_tool._canonical_bytes(receipt_tool._seal(payload)))


def test_receipt_is_generated_and_verified_with_two_independent_builds(tmp_path: Path) -> None:
    receipt_path = _generate(tmp_path)

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "ch7-evidence-build-receipt.v1"
    assert payload["package"]["admission_status"] == "not_admitted"
    assert payload["build"]["determinism"]["verified"] is True
    assert payload["build"]["independent_builds"][0] == {
        **payload["build"]["independent_builds"][1],
        "ordinal": 1,
    }
    assert payload["verification"]["receipt_created"] is False

    result = receipt_tool.verify_receipt(
        receipt_path,
        scratch_root=tmp_path / "verify-scratch",
    )
    assert result["status"] == "verified"
    assert result["admission_status"] == "not_admitted"
    assert result["independent_output_tree_sha256"] == payload["package"]["payload_tree_sha256"]


def test_receipt_schema_and_non_circular_integrity_contract(tmp_path: Path) -> None:
    receipt_path = _generate(tmp_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    schema = json.loads(receipt_tool.RECEIPT_SCHEMA.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(payload)
    assert payload["integrity"]["excluded_json_pointer"] == "#/integrity"
    assert payload["integrity"]["receipt_sha256"] == receipt_tool._payload_hash(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["build"]["tool_sources"]["builder"].update(sha256="0" * 64),
            "build.tool_sources.builder",
        ),
        (
            lambda payload: payload["source"].update(v2_config_sha256="0" * 64),
            "source",
        ),
        (
            lambda payload: payload["build"]["environment"]["dependency_identity"].update(
                lockfile_sha256="0" * 64
            ),
            "build.environment",
        ),
        (
            lambda payload: payload["build"]["independent_builds"][0].update(
                output_tree_sha256="0" * 64
            ),
            "build.independent_builds",
        ),
    ],
)
def test_receipt_fails_closed_on_stale_or_changed_bindings(
    tmp_path: Path, mutate: object, message: str
) -> None:
    receipt_path = _generate(tmp_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    mutate(mutated)
    _write_sealed(receipt_path, mutated)

    with pytest.raises(receipt_tool.Ch7EvidenceBuildReceiptError, match=message):
        receipt_tool.verify_receipt(
            receipt_path,
            scratch_root=tmp_path / "verify-scratch",
        )


def test_receipt_self_hash_rejects_unsealed_changes(tmp_path: Path) -> None:
    receipt_path = _generate(tmp_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["package"]["manifest_sha256"] = "0" * 64
    receipt_path.write_bytes(receipt_tool._canonical_bytes(payload))

    with pytest.raises(receipt_tool.Ch7EvidenceBuildReceiptError, match="self-hash"):
        receipt_tool.verify_receipt(
            receipt_path,
            scratch_root=tmp_path / "verify-scratch",
        )
