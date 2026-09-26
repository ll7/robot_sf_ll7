"""Focused contract tests for the published release-row bundle loader."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

import robot_sf.analysis_workbench.release_row_bundle as bundle_loader
from robot_sf.analysis_workbench.release_row_bundle import (
    ReleaseRowBundleError,
    load_release_rows,
)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, allow_nan=True, sort_keys=True).encode() + b"\n" for row in rows
    )


def _write_bundle(tmp_path: Path, *, rows_by_arm: dict[str, list[dict[str, Any]]]) -> Path:
    root = tmp_path / "bundle"
    entries: list[dict[str, Any]] = []
    for arm, rows in rows_by_arm.items():
        relative = Path("payload") / "runs" / f"{arm}__differential_drive" / "episodes.jsonl"
        raw = _jsonl(rows)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        entries.append(
            {
                "path": relative.relative_to("payload").as_posix(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "kind": "episodes",
            }
        )
    (root / "publication_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "benchmark-publication-bundle.v2",
                "files": sorted(entries, key=lambda item: item["path"]),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


def _archive_bundle(root: Path, target: Path) -> Path:
    with tarfile.open(target, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    return target


def _manifest_for_episode(
    raw: bytes,
    *,
    arm: str = "goal",
    manifest_path: str | None = None,
) -> dict[str, object]:
    """Build the smallest v2 manifest binding one episode member."""
    path = manifest_path or f"runs/{arm}__differential_drive/episodes.jsonl"
    return {
        "schema_version": "benchmark-publication-bundle.v2",
        "files": [
            {
                "path": path,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "kind": "episodes",
            }
        ],
    }


def _write_raw_bundle(
    tmp_path: Path,
    raw: bytes,
    *,
    manifest: dict[str, object] | None = None,
    arm: str = "goal",
    relative_episode: str | None = None,
) -> Path:
    """Write a raw JSONL member and an optionally overridden manifest."""
    root = tmp_path / "bundle"
    relative = relative_episode or f"payload/runs/{arm}__differential_drive/episodes.jsonl"
    episode_path = root / relative
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path.write_bytes(raw)
    payload = manifest or _manifest_for_episode(raw, arm=arm)
    (root / "publication_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _write_manifest_only(tmp_path: Path, payload: object) -> Path:
    """Write a bundle root containing only a publication manifest."""
    root = tmp_path / "bundle"
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        manifest_bytes = payload
    else:
        manifest_bytes = json.dumps(payload).encode()
    (root / "publication_manifest.json").write_bytes(manifest_bytes)
    return root


def _archive_members(target: Path, members: list[tuple[str, bytes, str]]) -> Path:
    """Create a tiny archive with regular, directory, or symlink members."""
    with tarfile.open(target, "w:gz") as archive:
        for name, payload, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                info.size = 0
                archive.addfile(info)
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "outside"
                info.size = 0
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    return target


def test_load_release_rows_verifies_extracted_root_and_preserves_nonfinite_values(
    tmp_path: Path,
) -> None:
    root = _write_bundle(
        tmp_path,
        rows_by_arm={
            "goal": [{"episode_id": "goal-1", "metrics": {"legacy": float("inf")}}],
            "orca": [{"episode_id": "orca-1", "metrics": {"legacy": float("nan")}}],
        },
    )

    rows, source = load_release_rows(root)

    assert [row["_release_arm"] for row in rows] == ["goal", "orca"]
    assert all(row["_source_member"].startswith("payload/runs/") for row in rows)
    assert source["bundle_sha256"] is None
    assert source["planner_ids"] == ["goal", "orca"]
    assert source["row_count"] == 2
    assert (
        source["manifest_sha256"]
        == hashlib.sha256((root / "publication_manifest.json").read_bytes()).hexdigest()
    )
    assert rows[0]["metrics"]["legacy"] == float("inf")
    assert rows[1]["metrics"]["legacy"] != rows[1]["metrics"]["legacy"]


def test_load_release_rows_verifies_archive_and_bundle_digest(tmp_path: Path) -> None:
    root = _write_bundle(
        tmp_path,
        rows_by_arm={"goal": [{"episode_id": "goal-1"}]},
    )
    archive = _archive_bundle(root, tmp_path / "bundle.tar.gz")

    rows, source = load_release_rows(archive)

    assert rows[0]["_release_arm"] == "goal"
    assert source["bundle_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert source["episode_members"] == ["payload/runs/goal__differential_drive/episodes.jsonl"]


@pytest.mark.parametrize("mutation", ["digest", "size", "malformed"])
def test_load_release_rows_rejects_unbound_or_malformed_episode_bytes(
    tmp_path: Path, mutation: str
) -> None:
    root = _write_bundle(tmp_path, rows_by_arm={"goal": [{"episode_id": "goal-1"}]})
    episode_path = root / "payload/runs/goal__differential_drive/episodes.jsonl"
    manifest_path = root / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "digest":
        manifest["files"][0]["sha256"] = "0" * 64
    elif mutation == "size":
        manifest["files"][0]["size_bytes"] += 1
    else:
        episode_path.write_text("{not-json}\n", encoding="utf-8")
        manifest["files"][0]["size_bytes"] = episode_path.stat().st_size
        manifest["files"][0]["sha256"] = hashlib.sha256(episode_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReleaseRowBundleError):
        load_release_rows(root)


def test_load_release_rows_rejects_duplicate_episode_ids_within_an_arm(tmp_path: Path) -> None:
    root = _write_bundle(
        tmp_path,
        rows_by_arm={
            "goal": [
                {"episode_id": "duplicate"},
                {"episode_id": "duplicate"},
            ]
        },
    )

    with pytest.raises(ReleaseRowBundleError, match="duplicate episode_id"):
        load_release_rows(root)


def test_load_release_rows_rejects_archive_path_escape(tmp_path: Path) -> None:
    manifest = json.dumps(
        {
            "schema_version": "benchmark-publication-bundle.v2",
            "files": [],
        }
    ).encode()
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in (
            ("bundle/publication_manifest.json", manifest),
            ("bundle/payload/runs/goal__differential_drive/episodes.jsonl", b"{}\n"),
            ("bundle/../outside.txt", b"escape"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ReleaseRowBundleError, match="unsafe path"):
        load_release_rows(archive_path)


@pytest.mark.parametrize(
    "mutation",
    ["schema", "files", "entry", "path", "size", "sha256", "duplicate"],
)
def test_load_release_rows_rejects_bad_manifest_entries(tmp_path: Path, mutation: str) -> None:
    """Publication metadata must bind one safe, checksummed episode member."""

    raw = b'{"episode_id":"goal-1"}\n'
    manifest = _manifest_for_episode(raw)
    entries = manifest["files"]
    if mutation == "schema":
        manifest["schema_version"] = "unknown"
    elif mutation == "files":
        manifest["files"] = {}
    elif mutation == "entry":
        manifest["files"] = [None]
    elif mutation == "path":
        entries[0]["path"] = "../escape/episodes.jsonl"
    elif mutation == "size":
        entries[0]["size_bytes"] = -1
    elif mutation == "sha256":
        entries[0]["sha256"] = "invalid"
    else:
        manifest["files"] = [entries[0], dict(entries[0])]
    root = _write_raw_bundle(tmp_path, raw, manifest=manifest)

    with pytest.raises(ReleaseRowBundleError):
        load_release_rows(root)


@pytest.mark.parametrize(
    "raw",
    [b"", b"\n", b"[]\n", b"{}\n", b'{"episode_id":"a","episode_id":"b"}\n'],
)
def test_load_release_rows_rejects_bad_episode_rows(tmp_path: Path, raw: bytes) -> None:
    """Empty, non-object, missing-ID, and ambiguous rows fail admission."""

    root = _write_raw_bundle(tmp_path, raw)

    with pytest.raises(ReleaseRowBundleError):
        load_release_rows(root)


def test_load_release_rows_rejects_missing_extracted_episode(tmp_path: Path) -> None:
    """A manifest without a corresponding row file cannot be admitted."""

    root = _write_manifest_only(tmp_path, _manifest_for_episode(b'{"episode_id":"a"}\n'))

    with pytest.raises(ReleaseRowBundleError, match="payload/runs"):
        load_release_rows(root)


def test_load_release_rows_rejects_unlisted_extracted_episode(tmp_path: Path) -> None:
    """An extra release arm cannot bypass the manifest's member set."""

    root = _write_bundle(tmp_path, rows_by_arm={"goal": [{"episode_id": "goal-1"}]})
    extra = root / "payload/runs/orca__differential_drive/episodes.jsonl"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b'{"episode_id":"orca-1"}\n')

    with pytest.raises(ReleaseRowBundleError, match="do not match bundle files"):
        load_release_rows(root)


def test_load_release_rows_rejects_symlink_in_extracted_runs(tmp_path: Path) -> None:
    """A symlink anywhere under payload/runs invalidates an extracted source."""

    root = _write_bundle(tmp_path, rows_by_arm={"goal": [{"episode_id": "goal-1"}]})
    (root / "payload/runs/outside").symlink_to(tmp_path)

    with pytest.raises(ReleaseRowBundleError, match="contains a symlink"):
        load_release_rows(root)


def test_load_release_rows_rejects_archive_symlink_member(tmp_path: Path) -> None:
    """Archive links cannot redirect an episode source outside the bundle."""

    raw = b'{"episode_id":"a"}\n'
    archive_path = _archive_members(
        tmp_path / "linked.tar.gz",
        [
            (
                "bundle/publication_manifest.json",
                json.dumps(_manifest_for_episode(raw)).encode(),
                "file",
            ),
            ("bundle/payload/runs/goal__differential_drive/episodes.jsonl", raw, "symlink"),
        ],
    )

    with pytest.raises(ReleaseRowBundleError, match="non-regular member"):
        load_release_rows(archive_path)


def test_load_release_rows_rejects_archive_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive byte cap is checked before opening or expanding members."""

    root = _write_bundle(tmp_path, rows_by_arm={"goal": [{"episode_id": "goal-1"}]})
    archive = _archive_bundle(root, tmp_path / "bundle.tar.gz")
    monkeypatch.setattr(bundle_loader, "MAX_ARCHIVE_BYTES", archive.stat().st_size - 1)

    with pytest.raises(ReleaseRowBundleError, match="input size limit"):
        load_release_rows(archive)
