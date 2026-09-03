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
    assert result["retrieval"]["rights"] == registry.RIGHTS_METADATA
    v1 = next(item for item in result["durable_sources"] if item["package_id"].endswith("-v1"))
    assert v1["member_sha256sums"]["manifest.json"] == (
        "39885f8a5abcd5acb1d02db9fa51ea03fd914460415babe0ede0a7744d9a35d1"
    )


def test_missing_registry_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(registry.SourceRegistryUnavailableError, match="unavailable"):
        registry.verify_source_registry(registry_path=tmp_path / "missing.json")


def test_invalid_registry_encoding_is_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_bytes(b"\xff")

    with pytest.raises(registry.SourceRegistryUnavailableError, match="unavailable"):
        registry.verify_source_registry(registry_path=path)


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


def test_repository_commit_must_resolve_to_a_commit_object(tmp_path: Path) -> None:
    payload = copy.deepcopy(_registry_payload())
    records = payload["durable_records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    repository = record["repository"]
    assert isinstance(repository, dict)
    repository["commit"] = (
        registry._git_bytes(ROOT, ("rev-parse", f"{registry.SOURCE_COMMIT}^{{tree}}"), "test tree")
        .decode("ascii")
        .strip()
    )
    _reseal(payload)
    path = tmp_path / "registry.json"
    _write_registry(path, payload)

    with pytest.raises(registry.SourceRegistryBlockedError, match="expected 'commit'"):
        registry.verify_source_registry(registry_path=path)


def test_package_entries_must_remain_in_canonical_order(tmp_path: Path) -> None:
    payload = copy.deepcopy(_registry_payload())
    records = payload["durable_records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    packages = record["packages"]
    assert isinstance(packages, list)
    packages.reverse()
    _reseal(payload)
    path = tmp_path / "registry.json"
    _write_registry(path, payload)

    with pytest.raises(
        registry.SourceRegistryBlockedError, match="canonical v1 and v2 package order"
    ):
        registry.verify_source_registry(registry_path=path)


def test_package_id_binds_role_path_media_and_required_members(tmp_path: Path) -> None:
    payload = copy.deepcopy(_registry_payload())
    records = payload["durable_records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    packages = record["packages"]
    assert isinstance(packages, list)
    v1, v2 = packages
    assert isinstance(v1, dict)
    assert isinstance(v2, dict)
    for field in (
        "role",
        "path",
        "media_type",
        "required_members",
        "tree_sha",
        "sha256sums_path",
        "sha256sums_sha256",
        "member_sha256sums",
        "durable_uri",
    ):
        v1[field] = copy.deepcopy(v2[field])
    _reseal(payload)
    path = tmp_path / "registry.json"
    _write_registry(path, payload)

    with pytest.raises(registry.SourceRegistryBlockedError, match="authorized package path|role"):
        registry.verify_source_registry(registry_path=path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("issue", 6792), ("comment_id", 1), ("decision", "approve")],
)
def test_approval_metadata_must_match_authorized_decision(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = copy.deepcopy(_registry_payload())
    records = payload["durable_records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    approval = record["approval"]
    assert isinstance(approval, dict)
    approval[field] = value
    _reseal(payload)
    path = tmp_path / "registry.json"
    _write_registry(path, payload)

    with pytest.raises(registry.SourceRegistryBlockedError, match="approval is not authorized"):
        registry.verify_source_registry(registry_path=path)


def test_rights_metadata_does_not_claim_member_redistribution_clearance(tmp_path: Path) -> None:
    payload = copy.deepcopy(_registry_payload())
    records = payload["durable_records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    retrieval = record["retrieval"]
    assert isinstance(retrieval, dict)
    rights = retrieval["rights"]
    assert isinstance(rights, dict)
    rights["status"] = "public_repository_history"
    _reseal(payload)
    path = tmp_path / "registry.json"
    _write_registry(path, payload)

    with pytest.raises(registry.SourceRegistryBlockedError, match="rights metadata"):
        registry.verify_source_registry(registry_path=path)


def test_nul_in_package_path_is_blocked(tmp_path: Path) -> None:
    payload = copy.deepcopy(_registry_payload())
    records = payload["durable_records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    packages = record["packages"]
    assert isinstance(packages, list)
    package = packages[0]
    assert isinstance(package, dict)
    package["path"] = f"{package['path']}\x00"
    _reseal(payload)
    path = tmp_path / "registry.json"
    _write_registry(path, payload)

    with pytest.raises(registry.SourceRegistryBlockedError, match="safe repository path"):
        registry.verify_source_registry(registry_path=path)


@pytest.mark.parametrize("package_id", [None, [], {}])
def test_malformed_package_id_is_blocked(tmp_path: Path, package_id: object) -> None:
    """Malformed package identity cannot escape as an incidental Python error."""
    payload = copy.deepcopy(_registry_payload())
    records = payload["durable_records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    packages = record["packages"]
    assert isinstance(packages, list)
    first_package = packages[0]
    assert isinstance(first_package, dict)
    first_package["package_id"] = package_id
    _reseal(payload)
    path = tmp_path / "registry.json"
    _write_registry(path, payload)

    with pytest.raises(registry.SourceRegistryBlockedError, match="malformed package ID"):
        registry.verify_source_registry(registry_path=path)
