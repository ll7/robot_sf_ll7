"""Fail-closed tests for the commit-pinned Chapter 7 source registry."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.analysis import verify_ch7_source_registry as registry

ROOT = Path(__file__).parents[2]
REGISTRY_PATH = ROOT / registry.REGISTRY_PATH


def _registry_payload() -> dict[str, object]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _write_registry(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _reseal(payload: dict[str, object]) -> None:
    records = payload["durable_records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    record["record_sha256"] = registry.record_sha256(record)


def test_valid_registry_retrieves_both_pinned_package_trees() -> None:
    result = registry.verify_source_registry()

    assert result["registry_id"] == registry.REGISTRY_ID
    assert result["record_id"] == registry.RECORD_ID
    assert result["source_commit"] == "a1892cf453973cd19e7bbba158a9f4132009bcee"
    assert {item["package_id"] for item in result["durable_sources"]} == {
        "ch7-evidence-package-v1",
        "ch7-evidence-package-v2",
    }
    assert result["retrieval"]["protocol"] == registry.RETRIEVAL_PROTOCOL


def test_missing_registry_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(registry.SourceRegistryUnavailableError, match="unavailable"):
        registry.verify_source_registry(registry_path=tmp_path / "missing.json")


def test_mismatched_member_digest_is_rejected(tmp_path: Path) -> None:
    payload = copy.deepcopy(_registry_payload())
    records = payload["durable_records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    packages = record["packages"]
    assert isinstance(packages, list)
    v2 = packages[1]
    assert isinstance(v2, dict)
    member_hashes = v2["member_sha256sums"]
    assert isinstance(member_hashes, dict)
    member_hashes["manifest.json"] = "0" * 64
    _reseal(payload)
    path = tmp_path / "registry.json"
    _write_registry(path, payload)

    with pytest.raises(registry.SourceRegistryMismatchError, match="member SHA-256 map"):
        registry.verify_source_registry(registry_path=path)


def test_inaccessible_pinned_repository_is_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    _write_registry(path, _registry_payload())

    with pytest.raises(registry.SourceRegistryUnavailableError, match="git object is unavailable"):
        registry.verify_source_registry(
            repository_root=tmp_path / "not-a-checkout",
            registry_path=path,
        )


def test_moving_commit_reference_is_blocked(tmp_path: Path) -> None:
    payload = copy.deepcopy(_registry_payload())
    records = payload["durable_records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    repository = record["repository"]
    assert isinstance(repository, dict)
    repository["commit"] = "main"
    _reseal(payload)
    path = tmp_path / "registry.json"
    _write_registry(path, payload)

    with pytest.raises(registry.SourceRegistryBlockedError, match="immutable"):
        registry.verify_source_registry(registry_path=path)
