"""Focused safety and identity tests for the generic dependency collector."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import urllib.request
import zipfile
from typing import TYPE_CHECKING

import pytest

from scripts.tools.collect_dependency_archive_evidence import (
    MAX_ARCHIVE_BYTES,
    MAX_EVIDENCE_MEMBER_BYTES,
    MAX_MANIFEST_MEMBER_COUNT,
    MAX_METADATA_JSON_BYTES,
    _CanonicalRedirectHandler,
    _collection_status,
    _pypi_archive_url,
    _pypi_exact_file_match,
    _pypi_json_url,
    _validate_manifest,
    collect,
    fetch_archive,
    fetch_json,
    inspect_archive,
    read_bounded_bytes,
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
        "url": f"https://files.pythonhosted.org/packages/{filename}",
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


def test_manifest_input_read_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b"1234")

    with pytest.raises(ValueError, match="batch manifest exceeds 3 byte bound"):
        read_bounded_bytes(path, 3, "batch manifest")


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


def test_manifest_validation_rejects_untyped_or_oversized_member_count() -> None:
    manifest = _manifest()
    manifest["member_count"] = True

    with pytest.raises(ValueError, match="member_count must be a typed integer"):
        _validate_manifest(manifest, _args())

    manifest = _manifest()
    manifest["member_count"] = MAX_MANIFEST_MEMBER_COUNT + 1

    with pytest.raises(ValueError, match="bounded collector limit"):
        _validate_manifest(manifest, _args())


def test_manifest_validation_rejects_normalized_name_mismatch() -> None:
    manifest = _manifest()
    manifest["members"][0]["normalized_name"] = "other"  # type: ignore[index]

    with pytest.raises(ValueError, match="normalized_name"):
        _validate_manifest(manifest, _args())


def test_manifest_validation_binds_selected_profile_and_archive_url_filename() -> None:
    manifest = _manifest()
    manifest["members"][0]["profiles"] = ["core"]  # type: ignore[index]

    with pytest.raises(ValueError, match="profiles do not include"):
        _validate_manifest(manifest, _args())

    manifest = _manifest()
    manifest["members"][0]["artifacts"][0]["url"] = (  # type: ignore[index]
        "https://files.pythonhosted.org/packages/other.whl"
    )

    with pytest.raises(ValueError, match="URL path does not identify filename"):
        _validate_manifest(manifest, _args())

    manifest = _manifest()
    manifest["members"][0]["artifacts"][0]["kind"] = "sdist"  # type: ignore[index]

    with pytest.raises(ValueError, match="sdist filename cannot end with"):
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


def test_target_selection_binds_platform_family_and_python_tag() -> None:
    manylinux = _artifact()
    assert target_compatible(manylinux, "Linux", "x86_64", "3.13")
    assert not target_compatible(manylinux, "Windows", "x86_64", "3.13")
    assert not target_compatible(manylinux, "Darwin", "x86_64", "3.13")

    windows = _artifact(filename="demo-1.0.0-py3-none-win_amd64.whl")
    windows["platform_tags"] = ["py3", "none", "win_amd64"]
    assert target_compatible(windows, "Windows", "amd64", "3.13")
    assert not target_compatible(windows, "Linux", "amd64", "3.13")

    macos = _artifact(filename="demo-1.0.0-py313-none-macosx_11_0_x86_64.whl")
    macos["platform_tags"] = ["py313", "none", "macosx_11_0_x86_64"]
    assert target_compatible(macos, "Darwin", "x86_64", "3.13")
    assert not target_compatible(macos, "Windows", "x86_64", "3.13")
    assert not target_compatible(macos, "Other", "x86_64", "3.13")


@pytest.mark.parametrize(
    ("validator", "url"),
    [
        (_pypi_json_url, "https://evil.example/pypi/demo/1.0/json"),
        (_pypi_archive_url, "https://evil.example/packages/demo.whl"),
    ],
)
def test_redirect_handler_rejects_noncanonical_hops(validator, url: str) -> None:
    handler = _CanonicalRedirectHandler(validator, "redirect rejected")
    request = urllib.request.Request("https://pypi.org/pypi/demo/1.0/json")

    with pytest.raises(ValueError, match="redirect rejected"):
        handler.redirect_request(request, None, 302, "Found", {}, url)


def test_pypi_exact_file_match_binds_distribution_kind() -> None:
    artifact = {
        "kind": "sdist",
        "filename": "demo-1.0.0.zip",
        "sha256": "a" * 64,
        "size": 10,
        "url": "https://files.pythonhosted.org/packages/demo-1.0.0.zip",
    }
    release_file = {
        "filename": artifact["filename"],
        "packagetype": "bdist_wheel",
        "digests": {"sha256": artifact["sha256"]},
        "size": artifact["size"],
        "url": artifact["url"],
    }

    assert not _pypi_exact_file_match(release_file, artifact)
    release_file["packagetype"] = "sdist"
    assert _pypi_exact_file_match(release_file, artifact)


def test_inspect_archive_rejects_over_cap_before_hashing(tmp_path: Path, monkeypatch) -> None:
    archive_path = tmp_path / "demo-1.0.0.whl"
    archive_path.write_bytes(b"not inspected")

    def fail_hash(*args: object, **kwargs: object) -> str:
        raise AssertionError("over-cap archive was hashed")

    monkeypatch.setattr("scripts.tools.collect_dependency_archive_evidence.sha256_file", fail_hash)
    result = inspect_archive(
        archive_path,
        {
            "filename": archive_path.name,
            "kind": "wheel",
            "url": "https://files.pythonhosted.org/packages/demo.whl",
            "size": MAX_ARCHIVE_BYTES + 1,
            "sha256": "a" * 64,
        },
    )

    assert result["digest_verified"] is False
    assert result["errors"] == ["archive manifest size exceeds or violates the bound"]


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


def test_evidence_member_bound_marks_archive_partial(tmp_path: Path) -> None:
    archive_path = tmp_path / "demo-1.0.0.whl"
    oversized_license = b"license\n" + b"x" * MAX_EVIDENCE_MEMBER_BYTES
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("LICENSE", oversized_license)
    data = archive_path.read_bytes()
    artifact = {
        "filename": archive_path.name,
        "kind": "wheel",
        "url": "https://files.pythonhosted.org/packages/demo-1.0.0.whl",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

    result = inspect_archive(
        archive_path,
        artifact,
        expected_name="demo",
        expected_version="1.0.0",
    )

    assert result["digest_verified"] is True
    assert result["inspection_status"] == "partial"
    assert result["evidence_members"][0]["read_status"].startswith("too_large>")
    assert any("could not be read" in error for error in result["errors"])


def test_zip_sdist_is_inspected_and_metadata_identity_is_bound(tmp_path: Path) -> None:
    archive_path = tmp_path / "demo-1.0.0.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "demo-1.0.0/PKG-INFO",
            "Metadata-Version: 2.3\nName: demo\nVersion: 1.0.0\n",
        )
        archive.writestr("demo-1.0.0/LICENSE", "Permission is granted.\n")
    data = archive_path.read_bytes()
    artifact = {
        "filename": archive_path.name,
        "kind": "sdist",
        "url": "https://files.pythonhosted.org/packages/demo-1.0.0.zip",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

    result = inspect_archive(
        archive_path,
        artifact,
        expected_name="demo",
        expected_version="1.0.0",
    )

    assert result["inspection_status"] == "inspected"
    assert result["metadata"][0]["fields"]["name"] == "demo"
    assert result["evidence_members"][0]["identity_match"] is True


def test_archive_metadata_identity_mismatch_is_partial(tmp_path: Path) -> None:
    archive_path = tmp_path / "demo-1.0.0.whl"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "demo-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.3\nName: another-demo\nVersion: 1.0.0\n",
        )
    data = archive_path.read_bytes()
    artifact = {
        "filename": archive_path.name,
        "kind": "wheel",
        "url": "https://files.pythonhosted.org/packages/demo-1.0.0.whl",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

    result = inspect_archive(
        archive_path,
        artifact,
        expected_name="demo",
        expected_version="1.0.0",
    )

    assert result["inspection_status"] == "partial"
    assert result["metadata"][0]["fields"]["name"] == "another-demo"
    assert result["evidence_members"][0]["identity_match"] is False
    assert any("identity differs" in error for error in result["errors"])


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


def test_archive_fetch_rejects_unapproved_https_host_without_touching_network(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_urlopen(*args: object, **kwargs: object) -> None:
        raise AssertionError("unapproved archive host attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    path, record, downloaded = fetch_archive(
        {
            "url": "https://files.example.test/demo.whl",
            "size": 1,
            "sha256": "a" * 64,
        },
        tmp_path / "demo.whl",
        0,
    )

    assert path is None
    assert record["error"] == "archive URL host is not an approved PyPI archive host"
    assert downloaded == 0


def test_offline_cache_misses_never_open_network(tmp_path: Path, monkeypatch) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline collection attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", fail_network)
    monkeypatch.setattr("urllib.request.build_opener", fail_network)

    payload, metadata_record = fetch_json(
        "https://pypi.org/pypi/demo/1.0.0/json",
        tmp_path / "metadata.json",
        offline=True,
    )
    path, archive_record, downloaded = fetch_archive(
        {
            "url": "https://files.pythonhosted.org/packages/demo.whl",
            "size": 1,
            "sha256": "a" * 64,
        },
        tmp_path / "demo.whl",
        0,
        offline=True,
    )

    assert payload is None
    assert "offline mode" in metadata_record["error"]
    assert path is None
    assert "offline mode" in archive_record["error"]
    assert downloaded == 0


def test_archive_failure_paths_are_redacted_from_ledger(tmp_path: Path, monkeypatch) -> None:
    artifact = _artifact()
    output = tmp_path / "output"
    manifest = _manifest(
        version=None,
        package_id=f"demo-package@editable#{IDENTITY_SHA[:16]}",
        source_type="editable",
        source={"editable": "."},
        artifacts=[artifact],
    )
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    args = _args(
        task_id="redacted-archive-failure",
        output=str(output),
        batch_manifest=str(manifest_path),
        expected_batch_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        target_os="Linux",
        target_architecture="x86_64",
        python_version="3.13",
        resolver_name="uv",
        resolver_version="0.11.21",
    )
    cache_path = output / "cache" / "demo_package" / "None" / artifact["filename"]

    class FailingOpener:
        def open(self, *args: object, **kwargs: object) -> None:
            raise OSError(f"cannot write {cache_path}")

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: FailingOpener())

    result = collect(args)

    ledger_text = (output / "dependency_evidence_ledger.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in ledger_text
    assert "<output>/cache/demo_package/None/" in ledger_text
    assert (
        "<output>/cache/demo_package/None/" in result["rows"][0]["archive_evidence"][0]["errors"][0]
    )


def test_archive_inspection_failure_paths_are_redacted(tmp_path: Path, monkeypatch) -> None:
    artifact = _artifact()
    output = tmp_path / "output"
    cache_path = output / "cache" / "demo_package" / "None" / artifact["filename"]
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"cached archive")
    artifact["size"] = cache_path.stat().st_size
    artifact["sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    manifest = _manifest(
        version=None,
        package_id=f"demo-package@editable#{IDENTITY_SHA[:16]}",
        source_type="editable",
        source={"editable": "."},
        artifacts=[artifact],
    )
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    args = _args(
        task_id="redacted-inspection-failure",
        output=str(output),
        batch_manifest=str(manifest_path),
        expected_batch_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        target_os="Linux",
        target_architecture="x86_64",
        python_version="3.13",
        resolver_name="uv",
        resolver_version="0.11.21",
        offline=True,
    )

    def fake_inspect(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "metadata": [],
            "evidence_members": [],
            "errors": [f"cannot inspect {cache_path}"],
        }

    monkeypatch.setattr(
        "scripts.tools.collect_dependency_archive_evidence.inspect_archive", fake_inspect
    )
    result = collect(args)

    archive = result["rows"][0]["archive_evidence"][0]
    assert str(tmp_path) not in json.dumps(archive)
    assert archive["errors"] == [
        "cannot inspect <output>/cache/demo_package/None/" + artifact["filename"]
    ]


def test_malformed_registry_metadata_remains_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact()
    archive_path = tmp_path / "output" / "cache" / "demo_package" / "1.0.0" / artifact["filename"]
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "demo-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.3\nName: demo-package\nVersion: 1.0.0\n",
        )
        archive.writestr("LICENSE", "Permission is granted.\n")
    data = archive_path.read_bytes()
    artifact["size"] = len(data)
    artifact["sha256"] = hashlib.sha256(data).hexdigest()

    output = tmp_path / "output"
    pypi_path = output / "pypi-json" / "demo_package-1.0.0.json"
    pypi_path.parent.mkdir(parents=True)
    pypi_path.write_text(
        json.dumps(
            {
                "info": {
                    "name": "demo-package",
                    "version": "1.0.0",
                    "description": "contact@example.invalid",
                    "classifiers": None,
                    "project_urls": {
                        "homepage": "https://example.invalid/demo",
                        "unexpected": {"private": "value"},
                    },
                },
                "urls": [
                    {
                        "filename": artifact["filename"],
                        "digests": {"sha256": artifact["sha256"]},
                        "size": artifact["size"],
                        "url": artifact["url"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = _manifest(artifacts=[artifact])
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    args = _args(
        task_id="malformed-metadata",
        output=str(output),
        batch_manifest=str(manifest_path),
        expected_batch_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        target_os="Linux",
        target_architecture="x86_64",
        python_version="3.13",
        resolver_name="uv",
        resolver_version="0.11.21",
        offline=True,
        argv="synthetic malformed metadata",
    )
    stale_cache_path = output / "cache" / "stale" / "unrelated.bin"
    stale_cache_path.parent.mkdir(parents=True)
    stale_cache_path.write_bytes(b"unrelated stale cache")

    def fail_rglob(*args: object, **kwargs: object) -> object:
        raise AssertionError("cache summary must not recursively scan the cache")

    monkeypatch.setattr("pathlib.Path.rglob", fail_rglob)

    result = collect(args)
    row = result["rows"][0]

    assert row["pypi_evidence"]["errors"]
    assert row["collection_status"] == "partial_or_unavailable"
    assert row["license_facts"]["authority_boundary"] == "descriptive_observation_only"
    observed_info = row["pypi_evidence"]["metadata"]["info"]
    assert "description" not in observed_info
    assert observed_info["project_urls"] == {"homepage": "https://example.invalid/demo"}
    ledger_text = (output / "dependency_evidence_ledger.json").read_text(encoding="utf-8")
    run_text = (output / "collector-run.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in ledger_text
    assert str(tmp_path) not in run_text
    assert result["target"]["offline"] is True
    cache_summary = json.loads((output / "private_cache_summary.json").read_text(encoding="utf-8"))
    assert all(item["path"] != "cache/stale/unrelated.bin" for item in cache_summary["files"])
    first_ledger = ledger_text
    collect(args)
    assert (output / "dependency_evidence_ledger.json").read_text(encoding="utf-8") == first_ledger


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
