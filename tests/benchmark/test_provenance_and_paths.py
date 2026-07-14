"""Contract tests for library-owned external-data paths and provenance gates."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.data.external import paths, provenance

if TYPE_CHECKING:
    from pathlib import Path


_REQUIRED_GROUPS = {
    "trajectory": ("**/obsmat.txt", "**/*.vsp", "**/*.txt"),
    "license_or_readme": ("**/README*", "**/LICENSE*", "**/TERMS*"),
}


def _complete_manifest(*, asset_id: str = "eth-ucy") -> dict[str, object]:
    """Return a compact manifest that satisfies the library readiness contract."""

    return {
        "asset_id": asset_id,
        "source_url": "https://example.invalid/eth-ucy",
        "license_note": "Licensed external data remains outside the repository.",
        "tree_sha256": "a" * 64,
        "sample_files": [{"path": "eth/obsmat.txt", "sha256": "b" * 64}],
        "matched_required_paths": ["eth/obsmat.txt", "README.txt"],
    }


def test_external_data_root_is_fail_closed_when_unset_or_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset and whitespace-only roots must not redirect external-data paths."""

    monkeypatch.delenv(paths.EXTERNAL_DATA_ROOT_ENV, raising=False)
    assert paths.external_data_root() is None

    monkeypatch.setenv(paths.EXTERNAL_DATA_ROOT_ENV, "   ")
    assert paths.external_data_root() is None


def test_external_data_path_resolves_shared_explicit_and_default_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The shared root, explicit root, and default path branches stay distinct."""

    default_path = tmp_path / "default" / "eth-ucy"
    shared_root = tmp_path / "shared"
    explicit_root = tmp_path / "explicit"

    monkeypatch.setenv(paths.EXTERNAL_DATA_ROOT_ENV, str(shared_root))
    assert paths.external_data_root() == shared_root.resolve()
    assert paths.resolve_external_data_path("eth-ucy", default_path) == shared_root / "eth-ucy"
    assert (
        paths.resolve_external_data_path("eth-ucy", default_path, root=explicit_root)
        == explicit_root / "eth-ucy"
    )

    monkeypatch.delenv(paths.EXTERNAL_DATA_ROOT_ENV)
    assert paths.resolve_external_data_path("eth-ucy", default_path) == default_path


def test_provenance_path_matching_covers_literal_glob_and_fail_closed_cases() -> None:
    """Required-path groups accept valid alternatives and reject malformed coverage."""

    assert not provenance._paths_cover_requirements(_REQUIRED_GROUPS, None)
    assert not provenance._paths_cover_requirements(_REQUIRED_GROUPS, [])
    assert not provenance._paths_cover_requirements(
        {"trajectory": ("required.txt",)}, ["unrelated.txt"]
    )
    assert provenance._paths_cover_requirements(
        {"trajectory": ("required.txt",)}, ["required.txt/nested/file"]
    )
    assert provenance._paths_cover_requirements({"trajectory": ("**/obsmat.txt",)}, ["obsmat.txt"])
    assert provenance._paths_cover_requirements(
        {"trajectory": ("**/*.txt",)}, ["nested/trajectory.txt"]
    )


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        (None, "missing_manifest"),
        ("{", "invalid_json"),
        (json.dumps(["not", "an", "object"]), "invalid_json"),
    ],
)
def test_provenance_manifest_rejects_missing_and_invalid_files(
    tmp_path: Path,
    payload: str | None,
    expected_status: str,
) -> None:
    """Missing, malformed, and non-object manifests fail closed with a status."""

    manifest_path = tmp_path / "manifest.json"
    if payload is not None:
        manifest_path.write_text(payload, encoding="utf-8")

    report = provenance.check_provenance_manifest(
        "eth-ucy",
        manifest_path,
        required_path_groups=_REQUIRED_GROUPS,
    )

    assert report["ok"] is False
    assert report["status"] == expected_status


def test_provenance_manifest_reports_all_incomplete_metadata(tmp_path: Path) -> None:
    """An incomplete object identifies metadata, checksum, and path omissions."""

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "asset_id": "other-asset",
                "source_url": " ",
                "license_note": None,
                "tree_sha256": "",
                "sample_files": [{"path": "eth/obsmat.txt"}],
                "matched_required_paths": ["eth/obsmat.txt"],
            }
        ),
        encoding="utf-8",
    )

    report = provenance.check_provenance_manifest(
        "eth-ucy",
        manifest_path,
        required_path_groups=_REQUIRED_GROUPS,
    )

    assert report["ok"] is False
    assert report["status"] == "incomplete_metadata"
    assert {
        "asset_id",
        "source_url",
        "license_note",
        "tree_sha256",
        "sample_files[].sha256",
        "matched_required_paths",
    }.issubset(report["missing_metadata"])


def test_provenance_manifest_accepts_complete_metadata(tmp_path: Path) -> None:
    """A manifest covering every required group is ready for the library loader."""

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_complete_manifest()), encoding="utf-8")

    report = provenance.check_provenance_manifest(
        "eth-ucy",
        manifest_path,
        required_path_groups=_REQUIRED_GROUPS,
    )

    assert report["ok"] is True
    assert report["status"] == "ready"
    assert report["missing_metadata"] == []
