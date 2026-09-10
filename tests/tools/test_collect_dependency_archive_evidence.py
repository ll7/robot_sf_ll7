"""Focused safety and identity tests for the generic dependency collector."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from typing import TYPE_CHECKING

import pytest

from scripts.tools.collect_dependency_archive_evidence import (
    MAX_ARCHIVE_BYTES,
    MAX_METADATA_JSON_BYTES,
    _collection_status,
    _validate_manifest,
    collect,
    fetch_archive,
    fetch_json,
    inspect_archive,
    select_target_artifacts,
    target_compatible,
)

if TYPE_CHECKING:
    from pathlib import Path

SOURCE_SHA = "1" * 40
CANDIDATE_SHA = "2" * 40
TREE_SHA = "3" * 40
IDENTITY_SHA = "a" * 64


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "expected_profile": "all",
        "expected_owner_issue": 8179,
        "expected_audit_source_sha": SOURCE_SHA,
        "expected_candidate_commit_sha": CANDIDATE_SHA,
        "expected_candidate_tree_sha": TREE_SHA,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _artifact(
    *, filename: str = "demo-1.0.0-py3-none-manylinux_2_17_x86_64.whl"
) -> dict[str, object]:
    return {
        "filename": filename,
        "kind": "wheel",
        "platform_tags": ["py3", "none", "manylinux_2_17_x86_64"],
        "sha256": "b" * 64,
        "size": 10,
        "url": f"https://files.example.test/{filename}",
    }


def _member(**overrides: object) -> dict[str, object]:
    member = {
        "name": "demo-package",
        "normalized_name": "demo-package",
        "version": "1.0.0",
        "package_id": f"demo-package@1.0.0#{IDENTITY_SHA[:16]}",
        "identity_sha256": IDENTITY_SHA,
        "lockfile": "uv.lock",
        "assignment": {
            "owner_issue": 8179,
            "owner_url": "https://github.com/ll7/robot_sf_ll7/issues/8179",
        },
        "selected_profiles": ["all"],
        "profiles": ["all"],
        "source_type": "registry",
        "source": {"registry": "https://pypi.org/simple"},
        "artifacts": [_artifact()],
        "distribution_mode": "user_installed",
        "surface_membership": "selected",
    }
    member.update(overrides)
    return member


def _manifest(**member_overrides: object) -> dict[str, object]:
    member = _member(**member_overrides)
    return {
        "schema_version": "robot_sf.p04.p05_batch.v1",
        "profile": "all",
        "owner_issue": 8179,
        "owner_url": "https://github.com/ll7/robot_sf_ll7/issues/8179",
        "audit_source_sha": SOURCE_SHA,
        "candidate_commit_sha": CANDIDATE_SHA,
        "candidate_tree_sha": TREE_SHA,
        "member_count": 1,
        "members": [member],
    }


def test_manifest_validation_binds_owner_profile_and_candidate_identities() -> None:
    assert _validate_manifest(_manifest(), _args()) == 8179


def test_manifest_validation_rejects_owner_mismatch_before_fetch() -> None:
    manifest = _manifest()
    manifest["members"][0]["assignment"]["owner_issue"] = 8172  # type: ignore[index]

    with pytest.raises(ValueError, match="assignment owner"):
        _validate_manifest(manifest, _args())


def test_manifest_validation_rejects_noncanonical_owner_url() -> None:
    manifest = _manifest()
    manifest["owner_url"] = "https://evil.example/issues/8179"

    with pytest.raises(ValueError, match="owner_url"):
        _validate_manifest(manifest, _args())


def test_manifest_validation_rejects_profile_mismatch_before_fetch() -> None:
    manifest = _manifest()
    manifest["profile"] = "minimal"

    with pytest.raises(ValueError, match="profile"):
        _validate_manifest(manifest, _args())


def test_manifest_validation_rejects_identity_suffix_mismatch() -> None:
    manifest = _manifest()
    manifest["members"][0]["package_id"] = (  # type: ignore[index]
        f"demo-package@1.0.0#{'c' * 16}"
    )

    with pytest.raises(ValueError, match="package_id suffix"):
        _validate_manifest(manifest, _args())


def test_manifest_validation_rejects_normalized_name_mismatch() -> None:
    manifest = _manifest()
    manifest["members"][0]["normalized_name"] = "other"  # type: ignore[index]

    with pytest.raises(ValueError, match="normalized_name"):
        _validate_manifest(manifest, _args())


def test_manifest_validation_rejects_lockfile_and_identity_hash_mismatch() -> None:
    manifest = _manifest(lockfile="requirements.txt")
    manifest["members"][0]["artifacts"][0]["sha256"] = "bad"  # type: ignore[index]

    with pytest.raises(ValueError, match="lockfile"):
        _validate_manifest(manifest, _args())

    manifest = _manifest()
    manifest["members"][0]["artifacts"][0]["sha256"] = "bad"  # type: ignore[index]

    with pytest.raises(ValueError, match="sha256"):
        _validate_manifest(manifest, _args())


def test_manifest_validation_rejects_unknown_source_and_unsafe_editable_path() -> None:
    manifest = _manifest(source_type="unknown")

    with pytest.raises(ValueError, match="source_type"):
        _validate_manifest(manifest, _args())

    manifest = _manifest(source_type="editable", source={"editable": "../outside"})

    with pytest.raises(ValueError, match="relative path"):
        _validate_manifest(manifest, _args())


def test_manifest_validation_rejects_noncanonical_registry_url() -> None:
    manifest = _manifest(source={"registry": "https://evil.example/simple"})

    with pytest.raises(ValueError, match="canonical PyPI"):
        _validate_manifest(manifest, _args())


def test_manifest_validation_rejects_duplicate_package_identity() -> None:
    manifest = _manifest()
    duplicate = dict(manifest["members"][0])  # type: ignore[index]
    manifest["members"] = [manifest["members"][0], duplicate]  # type: ignore[index]
    manifest["member_count"] = 2

    with pytest.raises(ValueError, match="duplicate package_id"):
        _validate_manifest(manifest, _args())


def test_target_selection_requires_python_and_rejects_ambiguous_wheels() -> None:
    assert target_compatible(_artifact(), "Linux", "x86_64", "3.13")
    cp313 = _artifact(filename="demo-1.0.0-cp313-cp313-manylinux_2_17_x86_64.whl")
    cp313["platform_tags"] = ["cp313", "cp313", "manylinux_2_17_x86_64"]
    assert not target_compatible(cp313, "Linux", "x86_64", "3.12")
    member = _member(
        artifacts=[
            _artifact(),
            _artifact(filename="demo-1.0.0-cp313-cp313-manylinux_2_17_x86_64.whl"),
        ]
    )

    with pytest.raises(ValueError, match="multiple target-compatible wheels"):
        select_target_artifacts(
            member, os_name="Linux", architecture="x86_64", python_version="3.13"
        )


def test_unsupported_target_artifacts_remain_unresolved() -> None:
    assert (
        _collection_status(
            selected=[],
            manifest_artifacts=[_artifact()],
            archive_complete=False,
            pypi_complete=True,
        )
        == "partial_or_unavailable"
    )
    assert (
        _collection_status(
            selected=[],
            manifest_artifacts=[],
            archive_complete=False,
            pypi_complete=True,
        )
        == "not_applicable_no_archive"
    )


def test_cached_metadata_bound_is_checked_before_read(tmp_path: Path) -> None:
    cache_path = tmp_path / "metadata.json"
    with cache_path.open("wb") as handle:
        handle.truncate(MAX_METADATA_JSON_BYTES + 1)

    payload, record = fetch_json("https://pypi.org/pypi/demo/1.0/json", cache_path)

    assert payload is None
    assert record["status"] == "unavailable"
    assert record["error"] == "metadata cache exceeds 16 MiB bound"


def test_over_cap_archive_emits_unavailable_row_ledger(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact["size"] = MAX_ARCHIVE_BYTES + 1
    member = _member(
        version=None,
        package_id=f"demo-package@editable#{IDENTITY_SHA[:16]}",
        source_type="editable",
        source={"editable": "."},
        artifacts=[artifact],
    )
    manifest = _manifest(
        version=None,
        package_id=f"demo-package@editable#{IDENTITY_SHA[:16]}",
        source_type="editable",
        source={"editable": "."},
        artifacts=[artifact],
    )
    manifest["members"] = [member]
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    args = _args()
    args.task_id = "over-cap"
    args.output = str(tmp_path / "output")
    args.batch_manifest = str(manifest_path)
    args.expected_batch_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    args.target_os = "Linux"
    args.target_architecture = "x86_64"
    args.python_version = "3.13"
    args.resolver_name = "uv"
    args.resolver_version = "0.11.21"
    args.argv = "synthetic over-cap"

    result = collect(args)
    row = result["rows"][0]

    assert (tmp_path / "output" / "dependency_evidence_ledger.json").is_file()
    assert row["collection_status"] == "partial_or_unavailable"
    assert row["archive_evidence"][0]["inspection_status"] == "unavailable"
    assert "per-archive bound" in row["archive_evidence"][0]["errors"][0]
    assert result["coverage"]["partial_or_unavailable_rows"] == 1


def test_zip_symlink_and_control_member_fail_closed(tmp_path: Path) -> None:
    archive_path = tmp_path / "demo.whl"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("demo-1.0.0.dist-info/METADATA", "Name: demo\nVersion: 1.0.0\n")
        symlink = zipfile.ZipInfo("demo-1.0.0.dist-info/licenses/link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(symlink, "target")
        symlink_dir = zipfile.ZipInfo("demo-1.0.0.dist-info/licenses/dir-link/")
        symlink_dir.create_system = 3
        symlink_dir.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(symlink_dir, "target")
        archive.writestr("bad\x00name", "bad")
    data = archive_path.read_bytes()
    artifact = {
        "filename": archive_path.name,
        "kind": "wheel",
        "url": "https://files.example.test/demo.whl",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

    result = inspect_archive(archive_path, artifact)

    assert result["digest_verified"] is True
    assert result["inspection_status"] == "partial"
    assert any("unsafe archive member" in error for error in result["errors"])


def test_archive_fetch_rejects_non_https_without_touching_network(
    tmp_path: Path,
) -> None:
    path, record, downloaded = fetch_archive(
        {"url": "http://files.example.test/demo.whl", "size": 1, "sha256": "a" * 64},
        tmp_path / "demo.whl",
        0,
    )

    assert path is None
    assert record["error"] == "archive URL must be HTTPS"
    assert downloaded == 0


def test_collect_continues_after_ambiguous_target_selection(tmp_path: Path) -> None:
    ambiguous_identity = "d" * 64
    ambiguous_artifacts = [
        _artifact(filename="ambiguous-1.0.0-py3-none-manylinux_2_17_x86_64.whl"),
        _artifact(filename="ambiguous-1.0.0-cp313-cp313-manylinux_2_17_x86_64.whl"),
    ]
    ambiguous = _member(
        name="ambiguous-package",
        normalized_name="ambiguous-package",
        package_id=f"ambiguous-package@1.0.0#{ambiguous_identity[:16]}",
        identity_sha256=ambiguous_identity,
        source_type="editable",
        source={"editable": "."},
        artifacts=ambiguous_artifacts,
    )

    ordinary_identity = "e" * 64
    ordinary_filename = "ordinary-1.0.0-py3-none-manylinux_2_17_x86_64.whl"
    ordinary_artifact = _artifact(filename=ordinary_filename)
    ordinary_cache = tmp_path / "output" / "cache" / "ordinary" / "1.0.0"
    ordinary_cache.mkdir(parents=True)
    ordinary_path = ordinary_cache / ordinary_filename
    with zipfile.ZipFile(ordinary_path, "w") as archive:
        archive.writestr(
            "ordinary-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.3\nName: ordinary\nVersion: 1.0.0\n",
        )
        archive.writestr("LICENSE", "Permission is granted.\n")
    ordinary_bytes = ordinary_path.read_bytes()
    ordinary_artifact["size"] = len(ordinary_bytes)
    ordinary_artifact["sha256"] = hashlib.sha256(ordinary_bytes).hexdigest()
    ordinary = _member(
        name="ordinary",
        normalized_name="ordinary",
        package_id=f"ordinary@1.0.0#{ordinary_identity[:16]}",
        identity_sha256=ordinary_identity,
        source_type="editable",
        source={"editable": "."},
        artifacts=[ordinary_artifact],
    )

    manifest = _manifest()
    manifest["member_count"] = 2
    manifest["members"] = [ambiguous, ordinary]
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    args = _args()
    args.task_id = "ambiguous-then-ordinary"
    args.output = str(tmp_path / "output")
    args.batch_manifest = str(manifest_path)
    args.expected_batch_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    args.target_os = "Linux"
    args.target_architecture = "x86_64"
    args.python_version = "3.13"
    args.resolver_name = "uv"
    args.resolver_version = "0.11.21"
    args.argv = "synthetic ambiguous row followed by ordinary row"

    result = collect(args)

    assert result["coverage"]["ledger_rows"] == 2
    assert result["coverage"]["exact_member_set"] is True
    assert result["coverage"]["complete_factual_rows"] == 1
    assert result["coverage"]["partial_or_unavailable_rows"] == 1
    assert result["coverage"]["unavailable_rows"] == [ambiguous["package_id"]]
    rows = result["rows"]
    assert [row["package_id"] for row in rows] == [
        ambiguous["package_id"],
        ordinary["package_id"],
    ]

    ambiguous_row, ordinary_row = rows
    assert ambiguous_row["collection_status"] == "partial_or_unavailable"
    assert ambiguous_row["target_context"]["selected_artifacts"] == []
    assert ambiguous_row["target_context"]["uninspected_manifest_artifacts"] == ambiguous_artifacts
    assert ambiguous_row["target_context"]["artifact_selection_status"] == "unresolved"
    selection_error = ambiguous_row["target_context"]["artifact_selection_error"]
    assert isinstance(selection_error, str)
    assert "multiple target-compatible wheels" in selection_error
    assert any(
        selection_error in question["question"]
        for question in ambiguous_row["unresolved_questions"]
    )

    assert ordinary_row["collection_status"] == "complete_factual_evidence"
    assert ordinary_row["target_context"]["artifact_selection_status"] == "resolved"
    assert ordinary_row["target_context"]["artifact_selection_error"] is None
    assert ordinary_row["archive_evidence"][0]["digest_verified"] is True
    assert ordinary_row["archive_evidence"][0]["inspection_status"] == "inspected"
