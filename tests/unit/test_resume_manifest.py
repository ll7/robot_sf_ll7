"""Tests for resume manifest invalidation rules (size, mtime, count, schema, identity).

The manifest sidecar now (v2) includes:
  - stat.size / stat.mtime_ns
  - episode_ids + episodes_count
  - schema_version
  - identity_hash (optional)

Invalidation conditions (each yields load_manifest -> None):
  1. size mismatch
  2. mtime mismatch
  3. episodes_count != len(episode_ids)
  4. schema_version mismatch
  5. identity_hash mismatch (when expected_identity_hash passed)
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

from robot_sf.benchmark.constants import EPISODE_SCHEMA_VERSION
from robot_sf.benchmark.manifest import load_manifest, save_manifest

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(tmp: Path, lines: list[dict]):
    with tmp.open("w", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")


def _basic_episode(eid: str) -> dict:
    return {"episode_id": eid}


def test_manifest_valid_roundtrip(tmp_path: Path):
    out = tmp_path / "episodes.jsonl"
    lines = [_basic_episode("sc1--0"), _basic_episode("sc1--1")]
    _write_jsonl(out, lines)
    save_manifest(out, [r["episode_id"] for r in lines], identity_hash="abc123")
    ids = load_manifest(out, expected_identity_hash="abc123")
    assert ids == {"sc1--0", "sc1--1"}


def test_invalidate_on_size_change(tmp_path: Path):
    out = tmp_path / "episodes.jsonl"
    lines = [_basic_episode("sc1--0")]
    _write_jsonl(out, lines)
    save_manifest(out, ["sc1--0"], identity_hash="h1")
    # Append new line without updating manifest
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_basic_episode("sc1--1")) + "\n")
    assert load_manifest(out, expected_identity_hash="h1") is None


def test_invalidate_on_mtime_change(tmp_path: Path):
    out = tmp_path / "episodes.jsonl"
    lines = [_basic_episode("sc1--0")]
    _write_jsonl(out, lines)
    save_manifest(out, ["sc1--0"], identity_hash="h1")
    # Touch the file without size change (rewrite same content but ensure mtime differs)
    time.sleep(0.01)  # buffer for coarse filesystems
    _write_jsonl(out, lines)  # same size, updated mtime via rewrite
    os.utime(out, None)  # explicitly bump mtime to guarantee stat change
    assert load_manifest(out, expected_identity_hash="h1") is None


def test_invalidate_on_schema_version_mismatch(tmp_path: Path):
    out = tmp_path / "episodes.jsonl"
    lines = [_basic_episode("sc1--0")]
    _write_jsonl(out, lines)
    # Write manifest with current helper, then manually patch schema_version to another value
    save_manifest(out, ["sc1--0"], identity_hash="h1")
    sidecar = out.with_suffix(out.suffix + ".manifest.json")
    data = json.loads(sidecar.read_text())
    data["schema_version"] = "vX"  # mismatched
    sidecar.write_text(json.dumps(data))
    assert (
        load_manifest(
            out,
            expected_identity_hash="h1",
            expected_schema_version=EPISODE_SCHEMA_VERSION,
        )
        is None
    )


def test_invalidate_on_episodes_count_mismatch(tmp_path: Path):
    out = tmp_path / "episodes.jsonl"
    lines = [_basic_episode("sc1--0"), _basic_episode("sc1--1")]
    _write_jsonl(out, lines)
    save_manifest(out, ["sc1--0", "sc1--1"], identity_hash="h1")
    sidecar = out.with_suffix(out.suffix + ".manifest.json")
    data = json.loads(sidecar.read_text())
    data["episodes_count"] = 999  # corrupt
    sidecar.write_text(json.dumps(data))
    assert load_manifest(out, expected_identity_hash="h1") is None


def test_invalidate_on_identity_hash_mismatch(tmp_path: Path):
    out = tmp_path / "episodes.jsonl"
    lines = [_basic_episode("sc1--0")]
    _write_jsonl(out, lines)
    save_manifest(out, ["sc1--0"], identity_hash="h1")
    assert load_manifest(out, expected_identity_hash="different") is None
