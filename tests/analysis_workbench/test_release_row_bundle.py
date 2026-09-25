"""Focused contract tests for the published release-row bundle loader."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

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
